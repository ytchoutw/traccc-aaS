#include <iostream>
#include <memory>

// CUDA include(s).
#include <cuda_runtime.h>

// Project include(s).
#include "traccc/clusterization/clustering_config.hpp"
#include "traccc/cuda/clusterization/clusterization_algorithm.hpp"
#include "traccc/cuda/clusterization/measurement_sorting_algorithm.hpp"
#include "traccc/cuda/clusterization/spacepoint_formation_algorithm.hpp"
#include "traccc/cuda/finding/finding_algorithm.hpp"
#include "traccc/cuda/fitting/fitting_algorithm.hpp"
#include "traccc/cuda/seeding/seeding_algorithm.hpp"
#include "traccc/cuda/seeding/track_params_estimation.hpp"
#include "traccc/cuda/utils/stream.hpp"
#include "traccc/edm/cell.hpp"
#include "traccc/edm/track_state.hpp"
#include "traccc/fitting/kalman_filter/kalman_fitter.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/device/container_d2h_copy_alg.hpp"

// Detray include(s).
#include "detray/core/detector.hpp"
#include "detray/detectors/bfield.hpp"
#include "detray/io/frontend/detector_reader.hpp"
#include "detray/navigation/navigator.hpp"
#include "detray/propagator/propagator.hpp"
#include "detray/propagator/rk_stepper.hpp"

// VecMem include(s).
#include <vecmem/containers/vector.hpp>
#include <vecmem/memory/binary_page_memory_resource.hpp>
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/memory_resource.hpp>
#include <vecmem/utils/cuda/async_copy.hpp>

// io
#include "traccc/io/read_cells.hpp"
#include "traccc/io/read_digitization_config.hpp"
#include "traccc/io/read_geometry.hpp"
#include "traccc/io/utils.hpp"
#include "traccc/io/csv/make_cell_reader.hpp"
#include "traccc/io/demonstrator_edm.hpp"
#include "traccc/io/read.hpp"

// clusterization
#include "traccc/options/accelerator.hpp"
#include "traccc/options/clusterization.hpp"
#include "traccc/options/detector.hpp"
#include "traccc/options/input_data.hpp"
#include "traccc/options/performance.hpp"
#include "traccc/options/program_options.hpp"
#include "traccc/options/program_options.hpp"
#include "traccc/options/track_finding.hpp"
#include "traccc/options/track_propagation.hpp"
#include "traccc/options/track_resolution.hpp"
#include "traccc/options/track_seeding.hpp"
#include "traccc/finding/finding_algorithm.hpp"
#include "traccc/fitting/fitting_algorithm.hpp"
#include "traccc/edm/cell.hpp"

// Command line option include(s).
#include "traccc/options/clusterization.hpp"
#include "traccc/options/detector.hpp"
#include "traccc/options/input_data.hpp"
#include "traccc/options/program_options.hpp"
#include "traccc/options/threading.hpp"
#include "traccc/options/throughput.hpp"
#include "traccc/options/track_finding.hpp"
#include "traccc/options/track_propagation.hpp"
#include "traccc/options/track_seeding.hpp"

// function to set the CUDA device and get the stream
static traccc::cuda::stream setCudaDeviceAndGetStream(int deviceID)
{
    cudaError_t err = cudaSetDevice(deviceID);
    if (err != cudaSuccess)
    {
        throw std::runtime_error("Failed to set CUDA device: \
                " + std::string(cudaGetErrorString(err)));
    }
    return traccc::cuda::stream(deviceID);
}

/// Helper macro for checking the return value of CUDA function calls
#define CUDA_ERROR_CHECK(EXP)                                                  \
    do {                                                                       \
        const cudaError_t errorCode = EXP;                                     \
        if (errorCode != cudaSuccess) {                                        \
            throw std::runtime_error(std::string("Failed to run " #EXP " (") + \
                                     cudaGetErrorString(errorCode) + ")");     \
        }                                                                      \
    } while (false)

// Definition of the cell_order struct
struct cell_order
{
    bool operator()(const traccc::cell &lhs, const traccc::cell &rhs) const
    {
        if (lhs.module_link != rhs.module_link)
        {
            return lhs.module_link < rhs.module_link;
        }
        else if (lhs.channel1 != rhs.channel1)
        {
            return lhs.channel1 < rhs.channel1;
        }
        else
        {
            return lhs.channel0 < rhs.channel0;
        }
    }
};

/// Helper function which finds module from csv::cell in the geometry and
/// digitization config, and initializes the modules limits with the cell's
/// properties
traccc::cell_module get_module(const std::uint64_t geometry_id,
                               const traccc::geometry* geom,
                               const traccc::digitization_config* dconfig,
                               const std::uint64_t original_geometry_id) {

    traccc::cell_module result;
    result.surface_link = detray::geometry::barcode{geometry_id};

    // Find/set the 3D position of the detector module.
    if (geom != nullptr) {

        // Check if the module ID is known.
        if (!geom->contains(result.surface_link.value())) {
            throw std::runtime_error(
                "Could not find placement for geometry ID " +
                std::to_string(result.surface_link.value()));
        }

        // Set the value on the module description.
        result.placement = (*geom)[result.surface_link.value()];
    }

    // Find/set the digitization configuration of the detector module.
    if (dconfig != nullptr) {

        // Check if the module ID is known.
        const traccc::digitization_config::Iterator geo_it =
            dconfig->find(original_geometry_id);
        if (geo_it == dconfig->end()) {
            throw std::runtime_error(
                "Could not find digitization config for geometry ID " +
                std::to_string(original_geometry_id));
        }

        // Set the value on the module description.
        const auto& binning_data = geo_it->segmentation.binningData();
        assert(binning_data.size() > 0);
        result.pixel.min_corner_x = binning_data[0].min;
        result.pixel.pitch_x = binning_data[0].step;
        if (binning_data.size() > 1) {
            result.pixel.min_corner_y = binning_data[1].min;
            result.pixel.pitch_y = binning_data[1].step;
        }
        result.pixel.dimension = geo_it->dimensions;
    }

    return result;
}

std::vector<traccc::io::csv::cell> read_csv(const std::string &filename);
std::map<std::uint64_t, std::vector<traccc::cell>> read_deduplicated_cells(const std::vector<traccc::io::csv::cell> &cells);
std::map<std::uint64_t, std::vector<traccc::cell>> read_all_cells(const std::vector<traccc::io::csv::cell> &cells);
void read_cells(traccc::io::cell_reader_output &out, 
                const std::vector<traccc::io::csv::cell> &cells,
                const traccc::geometry *geom, 
                const traccc::digitization_config *dconfig, 
                const std::map<std::uint64_t, detray::geometry::barcode> *barcode_map, 
                bool deduplicate);

// Type definitions
using host_detector_type = detray::detector<detray::default_metadata,
                                            detray::host_container_types>;
using device_detector_type =
    detray::detector<detray::default_metadata,
                        detray::device_container_types>;
using stepper_type =
    detray::rk_stepper<detray::bfield::const_field_t::view_t,
                        host_detector_type::algebra_type,
                        detray::constrained_step<>>;
using host_navigator_type = detray::navigator<const host_detector_type>;
using device_navigator_type = detray::navigator<const device_detector_type>;

// Tracking finding algorithm type
using host_finding_algorithm =
    traccc::finding_algorithm<stepper_type, host_navigator_type>;
using device_finding_algorithm =
    traccc::cuda::finding_algorithm<stepper_type, device_navigator_type>;

// Tracking fitting algorithm type
using host_fitting_algorithm = traccc::fitting_algorithm<
    traccc::kalman_fitter<stepper_type, host_navigator_type>>;
using device_fitting_algorithm = traccc::cuda::fitting_algorithm<
    traccc::kalman_fitter<stepper_type, device_navigator_type>>;

class TracccGpuStandalone
{
private:
    /// Device ID to use
    int m_device_id;

    /// Host memory resource
    vecmem::host_memory_resource m_host_mr;
    /// CUDA stream to use
    traccc::cuda::stream m_stream;
    /// Device memory resource
    vecmem::cuda::device_memory_resource m_device_mr;
    /// Device caching memory resource
    std::unique_ptr<vecmem::binary_page_memory_resource> m_cached_device_mr;
    /// (Asynchronous) memory copy object
    vecmem::cuda::async_copy m_copy;
    /// Memory resource for the host memory
    traccc::memory_resource m_mr;

    /// data configuration
    traccc::geometry m_surface_transforms;
    /// digitization configuration
    std::unique_ptr<traccc::digitization_config> m_digi_cfg;
    /// barcode map
    std::unique_ptr<std::map<std::uint64_t, detray::geometry::barcode>> m_barcode_map;

    // program configuration 
    // TODO: may need to be initialized!
    /// detector options
    traccc::opts::detector m_detector_opts;
    /// propagation options
    traccc::opts::track_propagation m_propagation_opts;
    /// clusterization options
    detray::propagation::config m_propagation_config;
    /// Configuration for clustering
    traccc::clustering_config m_clustering_config;
    /// Configuration for the seed finding
    traccc::seedfinder_config m_finder_config;
    /// Configuration for the spacepoint grid formation
    traccc::spacepoint_grid_config m_grid_config;
    /// Configuration for the seed filtering
    traccc::seedfilter_config m_filter_config;

    /// further configuration
    // TODO: may need to initialize differently / more explicitly
    detray::io::detector_reader_config m_cfg;
    /// Configuration for the track finding
    device_finding_algorithm::config_type m_finding_config;
    /// Configuration for the track fitting
    device_fitting_algorithm::config_type m_fitting_config;

    /// Constant B field for the (seed) track parameter estimation
    traccc::vector3 m_field_vec;
    /// Constant B field for the track finding and fitting
    detray::bfield::const_field_t m_field;

    /// Host detector
    host_detector_type* m_detector;
    /// Buffer holding the detector's payload on the device
    host_detector_type::buffer_type m_device_detector;
    /// View of the detector's payload on the device
    host_detector_type::view_type m_device_detector_view;

    /// Sub-algorithms used by this full-chain algorithm
    /// Clusterization algorithm
    traccc::cuda::clusterization_algorithm m_clusterization;
    /// Measurement sorting algorithm
    traccc::cuda::measurement_sorting_algorithm m_measurement_sorting;
    /// Spacepoint formation algorithm
    traccc::cuda::spacepoint_formation_algorithm m_spacepoint_formation;
    /// Seeding algorithm
    traccc::cuda::seeding_algorithm m_seeding;
    /// Track parameter estimation algorithm
    traccc::cuda::track_params_estimation m_track_parameter_estimation;

    /// Track finding algorithm
    device_finding_algorithm m_finding;
    /// Track fitting algorithm
    device_fitting_algorithm m_fitting;

    // copying to cpu
    traccc::device::container_d2h_copy_alg<
        traccc::track_candidate_container_types>
        m_copy_track_candidates;
    traccc::device::container_d2h_copy_alg<traccc::track_state_container_types>
        m_copy_track_states;

    // buffers
    traccc::cell_collection_types::buffer cells_buffer;
    traccc::cell_module_collection_types::buffer modules_buffer;

public:
    TracccGpuStandalone(std::vector<traccc::io::csv::cell> cells, int deviceID = 0) :
        m_device_id(deviceID), 
        m_host_mr(),
        m_stream(setCudaDeviceAndGetStream(deviceID)),
        m_device_mr(deviceID),
        m_cached_device_mr(
            std::make_unique<vecmem::binary_page_memory_resource>(m_device_mr)),
        m_copy(m_stream.cudaStream()),
        m_mr{*m_cached_device_mr, &m_host_mr},
        m_propagation_config(m_propagation_opts),
        m_clustering_config{256, 16, 8, 256},
        m_finder_config(), 
        m_grid_config(m_finder_config), 
        m_filter_config(), 
        m_finding_config(), 
        m_fitting_config(), 
        m_field_vec{0.f, 0.f, m_finder_config.bFieldInZ},
        m_field(detray::bfield::create_const_field(m_field_vec)),
        m_clusterization(m_mr, m_copy, m_stream, m_clustering_config),
        m_measurement_sorting(m_copy, m_stream),
        m_spacepoint_formation(m_mr, m_copy, m_stream),
        m_seeding(m_finder_config, m_grid_config, m_filter_config, 
                    m_mr, m_copy, m_stream),
        m_track_parameter_estimation(m_mr, m_copy, m_stream),
        m_finding(m_finding_config, m_mr, m_copy, m_stream),
        m_fitting(m_fitting_config, m_mr, m_copy, m_stream),
        m_copy_track_candidates(m_mr, m_copy),
        m_copy_track_states(m_mr, m_copy)
    {
        // Tell the user what device is being used.
        int device = 0;
        CUDA_ERROR_CHECK(cudaGetDevice(&device));
        cudaDeviceProp props;
        CUDA_ERROR_CHECK(cudaGetDeviceProperties(&props, device));
        std::cout << "Using CUDA device: " << props.name << " [id: " << device
                << ", bus: " << props.pciBusID
                << ", device: " << props.pciDeviceID << "]" << std::endl;

        initialize(cells);
    }

    // default destructor
    ~TracccGpuStandalone() = default;

    void initialize(std::vector<traccc::io::csv::cell> cells);
    void run();
    // std::vector<traccc::io::csv::cell> read_csv(const std::string &filename);
    std::vector<std::vector<double>> read_from_csv(const std::string &filename);
    std::vector<traccc::io::csv::cell> 
        read_from_array(const std::vector<std::vector<double>> &data);
};

void TracccGpuStandalone::initialize(std::vector<traccc::io::csv::cell> cells)
{
    
    // HACK: hard code location of detector and digitization file
    m_detector_opts.detector_file = "/global/cfs/projectdirs/m3443/data/traccc-aaS/data/geometries/odd/odd-detray_geometry_detray.json";
    m_detector_opts.digitization_file = "/global/cfs/projectdirs/m3443/data/traccc-aaS/data/geometries/odd/odd-digi-geometric-config.json";
    m_detector_opts.grid_file = "/global/cfs/projectdirs/m3443/data/traccc-aaS/data/geometries/odd/odd-detray_surface_grids_detray.json";
    m_detector_opts.use_detray_detector = true;

    // Read the digitization configuration file
    m_digi_cfg = std::make_unique<traccc::digitization_config>
        (traccc::io::read_digitization_config(m_detector_opts.digitization_file));

    // read in geometry
    auto geom_data = traccc::io::read_geometry(m_detector_opts.detector_file, traccc::data_format::json);
    m_surface_transforms = std::move(geom_data.first);
    m_barcode_map = std::move(geom_data.second);

    // setup the detector
    m_cfg.add_file(m_detector_opts.detector_file);
    m_cfg.add_file(m_detector_opts.grid_file);

    // Read the detector configuration file
    auto det = detray::io::read_detector<host_detector_type>(m_host_mr, m_cfg);
    m_detector = std::move(&det.first);

    // Copy it to the device.
    m_device_detector = detray::get_buffer(detray::get_data(*m_detector),
                                            m_device_mr, m_copy);
    m_stream.synchronize();
    m_device_detector_view = detray::get_data(m_device_detector);

    traccc::io::cell_reader_output read_out(m_mr.host);

    // Read the cells from the relevant event file into host memory.
    read_cells(read_out, cells, &m_surface_transforms, 
                m_digi_cfg.get(), m_barcode_map.get(), true);

    traccc::cell_collection_types::host& cells_per_event = read_out.cells;
    traccc::cell_module_collection_types::host& modules_per_event = read_out.modules;

    // create buffers and copy to device
    // Create device copy of input collections
    cells_buffer = vecmem::data::vector_buffer<traccc::cell>(cells_per_event.size(), *m_cached_device_mr);
    m_copy(vecmem::get_data(cells_per_event), cells_buffer)->ignore();
    modules_buffer = vecmem::data::vector_buffer<traccc::cell_module>(modules_per_event.size(), *m_cached_device_mr);
    m_copy(vecmem::get_data(modules_per_event), modules_buffer)->ignore();

    return;
}

void TracccGpuStandalone::run()
{
    //
    // ----------------- Clusterization -----------------
    // 
    const traccc::cuda::clusterization_algorithm::output_type measurements =
        m_clusterization(cells_buffer, modules_buffer);
    m_measurement_sorting(measurements);
    
    //
    // ----------------- Spacepoint Formation -----------------
    //  
    const traccc::cuda::spacepoint_formation_algorithm::output_type spacepoints =
        m_spacepoint_formation(measurements, modules_buffer);

    //
    // ----------------- Seeding and track param est. -----------
    //
    
    const traccc::cuda::seeding_algorithm::output_type seeds =
        m_seeding(spacepoints);

    const traccc::cuda::track_params_estimation::output_type track_params =
        m_track_parameter_estimation(spacepoints, seeds, m_field_vec);

    //
    // ----------------- Finding and Fitting -----------------
    //
    // track finding                        
    // Run the track finding (asynchronously).
    const device_finding_algorithm::output_type track_candidates = m_finding(
        m_device_detector_view, m_field, measurements, track_params);

    // Run the track fitting (asynchronously).
    const device_fitting_algorithm::output_type track_states = m_fitting(
        m_device_detector_view, m_field, track_candidates);

    // // Copy a limited amount of result data back to the host.
    // output_type result{&m_host_mr};
    // m_copy(track_states.headers, result)->wait();
    // return result;

    //
    // ----------------- Print Statistics -----------------
    // 
    // // copy buffer to host
    // traccc::measurement_collection_types::host measurements_per_event_cuda;
    // traccc::spacepoint_collection_types::host spacepoints_per_event_cuda;
    // traccc::seed_collection_types::host seeds_cuda;
    // traccc::bound_track_parameters_collection_types::host params_cuda;

    // m_copy(measurements, measurements_per_event_cuda)->wait();
    // m_copy(spacepoints, spacepoints_per_event_cuda)->wait();
    // m_copy(seeds, seeds_cuda)->wait();
    // m_copy(track_params, params_cuda)->wait();
    // auto track_candidates_cuda =
    //     m_copy_track_candidates(track_candidates);
    // auto track_states_cuda = m_copy_track_states(track_states);
    // m_stream.synchronize();

    // // print results
    // std::cout << " " << std::endl;
    // std::cout << "==> Statistics ... " << std::endl;
    // std::cout << " - number of measurements created " << measurements_per_event_cuda.size() << std::endl;
    // std::cout << " - number of spacepoints created " << spacepoints_per_event_cuda.size() << std::endl;
    // std::cout << " - number of seeds created " << seeds_cuda.size() << std::endl;
    // std::cout << " - number of track candidates created " << track_candidates_cuda.size() << std::endl;
    // std::cout << " - number of fitted tracks created " << track_states_cuda.size() << std::endl;
    std::cout << " done! " << std::endl;

    return;
}

// deal with input data

std::vector<traccc::io::csv::cell> read_csv(const std::string &filename)
{
    std::vector<traccc::io::csv::cell> cells;
    auto reader = traccc::io::csv::make_cell_reader(filename);
    traccc::io::csv::cell iocell;

    std::cout << "Reading cells from " << filename << std::endl;

    while (reader.read(iocell))
    {
        cells.push_back(iocell);
    }

    std::cout << "Read " << cells.size() << " cells." << std::endl;

    return cells;
}

std::vector<traccc::io::csv::cell> TracccGpuStandalone::read_from_array(const std::vector<std::vector<double>> &data)
{
    std::vector<traccc::io::csv::cell> cells;

    for (const auto &row : data)
    {
        if (row.size() != 6)
            continue; // ensure each row contains exactly 6 elements
        traccc::io::csv::cell iocell;
        // FIXME needs to decode to the correct type
        iocell.geometry_id = static_cast<std::uint64_t>(row[0]);
        iocell.hit_id = static_cast<int>(row[1]);
        iocell.channel0 = static_cast<int>(row[2]);
        iocell.channel1 = static_cast<int>(row[3]);
        iocell.timestamp = static_cast<int>(row[4]);
        iocell.value = row[5];
        cells.push_back(iocell);
    }

    return cells;
}

std::map<std::uint64_t, std::map<traccc::cell, float, cell_order>> fill_cell_map(const std::vector<traccc::io::csv::cell> &cells, unsigned int &nduplicates)
{
    std::map<std::uint64_t, std::map<traccc::cell, float, cell_order>> cellMap;
    nduplicates = 0;

    for (const auto &iocell : cells)
    {
        traccc::cell cell{iocell.channel0, iocell.channel1, iocell.value, iocell.timestamp, 0};
        auto ret = cellMap[iocell.geometry_id].insert({cell, iocell.value});
        if (!ret.second)
        {
            cellMap[iocell.geometry_id].at(cell) += iocell.value;
            ++nduplicates;
        }
    }

    return cellMap;
}

std::map<std::uint64_t, std::vector<traccc::cell>> create_result_container(const std::map<std::uint64_t, std::map<traccc::cell, float, cell_order>> &cellMap)
{
    std::map<std::uint64_t, std::vector<traccc::cell>> result;
    for (const auto &[geometry_id, cells] : cellMap)
    {
        for (const auto &[cell, value] : cells)
        {
            traccc::cell summed_cell{cell};
            summed_cell.activation = value;
            result[geometry_id].push_back(summed_cell);
        }
    }
    return result;
}

std::map<std::uint64_t, std::vector<traccc::cell>> read_deduplicated_cells(const std::vector<traccc::io::csv::cell> &cells)
{
    unsigned int nduplicates = 0;
    auto cellMap = fill_cell_map(cells, nduplicates);

    if (nduplicates > 0)
    {
        std::cout << "WARNING: " << nduplicates << " duplicate cells found." << std::endl;
    }

    return create_result_container(cellMap);
}

std::map<std::uint64_t, std::vector<traccc::cell>> read_all_cells(const std::vector<traccc::io::csv::cell> &cells)
{
    std::map<std::uint64_t, std::vector<traccc::cell>> result;

    for (const auto &iocell : cells)
    {
        traccc::cell cell{iocell.channel0, iocell.channel1, iocell.value, iocell.timestamp, 0};
        result[iocell.geometry_id].push_back(cell);
    }

    return result;
}

void read_cells(traccc::io::cell_reader_output &out, 
                const std::vector<traccc::io::csv::cell> &cells, 
                const traccc::geometry *geom, 
                const traccc::digitization_config *dconfig, 
                const std::map<std::uint64_t, detray::geometry::barcode> *barcode_map, 
                bool deduplicate)
{
    auto cellsMap = (deduplicate ? read_deduplicated_cells(cells)
                                 : read_all_cells(cells));

    for (const auto &[original_geometry_id, cells] : cellsMap)
    {
        std::uint64_t geometry_id = original_geometry_id;
        if (barcode_map != nullptr)
        {
            const auto it = barcode_map->find(geometry_id);
            if (it != barcode_map->end())
            {
                geometry_id = it->second.value();
            }
            else
            {
                throw std::runtime_error(
                    "Could not find barcode for geometry ID " +
                    std::to_string(geometry_id));
            }
        }

        out.modules.push_back(
            get_module(geometry_id, geom, dconfig, original_geometry_id));
        for (auto &cell : cells)
        {
            out.cells.push_back(cell);
            out.cells.back().module_link = out.modules.size() - 1;
        }
    }
}