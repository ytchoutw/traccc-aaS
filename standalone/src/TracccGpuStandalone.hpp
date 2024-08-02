#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>

#include <cuda_runtime.h>

// Project include(s).
#include "traccc/clusterization/clusterization_algorithm.hpp"
#include "traccc/cuda/clusterization/clusterization_algorithm.hpp"
#include "traccc/cuda/clusterization/measurement_sorting_algorithm.hpp"
#include "traccc/cuda/clusterization/spacepoint_formation_algorithm.hpp"
#include "traccc/cuda/utils/stream.hpp"

// io
#include "traccc/io/read_cells.hpp"
#include "traccc/io/read_digitization_config.hpp"
#include "traccc/io/read_geometry.hpp"
#include "traccc/io/utils.hpp"
#include "traccc/io/csv/make_cell_reader.hpp"

// clusterization
#include "traccc/options/accelerator.hpp"
#include "traccc/options/clusterization.hpp"
#include "traccc/options/detector.hpp"
#include "traccc/options/input_data.hpp"
#include "traccc/options/performance.hpp"
#include "traccc/options/program_options.hpp"
#include "traccc/edm/cell.hpp"

// #include "traccc/options/track_finding.hpp"
// #include "traccc/options/track_propagation.hpp"
// #include "traccc/options/track_seeding.hpp"
#include "traccc/performance/collection_comparator.hpp"
#include "traccc/performance/container_comparator.hpp"
#include "traccc/performance/timer.hpp"

// Detray include(s).
#include "detray/core/detector.hpp"
#include "detray/detectors/bfield.hpp"
#include "detray/io/frontend/detector_reader.hpp"

// VecMem include(s).
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/cuda/host_memory_resource.hpp>
#include <vecmem/memory/host_memory_resource.hpp>
#include <vecmem/utils/cuda/async_copy.hpp>

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

// Definition of the get_module function
traccc::cell_module get_module(const std::uint64_t geometry_id,
                               const traccc::geometry *geom,
                               const traccc::digitization_config *dconfig,
                               const std::uint64_t original_geometry_id)
{
    traccc::cell_module result;
    result.surface_link = detray::geometry::barcode{geometry_id};

    // Find/set the 3D position of the detector module.
    if (geom != nullptr)
    {
        if (!geom->contains(result.surface_link.value()))
        {
            throw std::runtime_error(
                "Could not find placement for geometry ID " +
                std::to_string(result.surface_link.value()));
        }
        result.placement = (*geom)[result.surface_link.value()];
    }

    // Find/set the digitization configuration of the detector module.
    if (dconfig != nullptr)
    {
        const traccc::digitization_config::Iterator geo_it =
            dconfig->find(original_geometry_id);
        if (geo_it == dconfig->end())
        {
            throw std::runtime_error(
                "Could not find digitization config for geometry ID " +
                std::to_string(original_geometry_id));
        }

        const auto &binning_data = geo_it->segmentation.binningData();
        assert(binning_data.size() > 0);
        result.pixel.min_corner_x = binning_data[0].min;
        result.pixel.pitch_x = binning_data[0].step;
        if (binning_data.size() > 1)
        {
            result.pixel.min_corner_y = binning_data[1].min;
            result.pixel.pitch_y = binning_data[1].step;
        }
        result.pixel.dimension = geo_it->dimensions;
        result.pixel.variance_y = geo_it->variance_y;
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

class TracccGpuStandalone
{
private:
    int m_device_id;
    // memory resources
    vecmem::host_memory_resource host_mr;
    vecmem::cuda::host_memory_resource cuda_host_mr;
    vecmem::cuda::device_memory_resource device_mr;
    traccc::memory_resource mr{device_mr, &cuda_host_mr};
    // CUDA types used
    traccc::cuda::stream stream;
    vecmem::cuda::async_copy copy{stream.cudaStream()};
    // opt inputs
    traccc::opts::detector detector_opts;
    traccc::opts::input_data input_opts;
    traccc::opts::clusterization clusterization_opts;
    traccc::opts::accelerator accelerator_opts;
    // detector options
    traccc::geometry surface_transforms;
    std::unique_ptr<traccc::digitization_config> digi_cfg;
    std::unique_ptr<std::map<std::uint64_t, detray::geometry::barcode>> barcode_map;
    detray::detector<detray::default_metadata,
        detray::host_container_types> host_detector{host_mr};
    detray::detector<detray::default_metadata,
        detray::host_container_types>::buffer_type device_detector;
    // detray::io::detector_reader_config cfg;
    // algorithms
    traccc::cuda::clusterization_algorithm ca_cuda;
    traccc::cuda::measurement_sorting_algorithm ms_cuda;
    traccc::measurement_collection_types::host measurements_per_event_cuda;

public:
    TracccGpuStandalone(int deviceID = 0) :
        // device_mr{deviceID},
        // mr{device_mr, &cuda_host_mr},
        // stream{deviceID},
        // copy{stream.cudaStream()},
        ca_cuda(mr, copy, stream, clusterization_opts), 
        ms_cuda(copy, stream)
    {
        // cudaSetDevice(deviceID);
        std::cout << "Device ID " << deviceID << std::endl;
        std::cout << "Current device: " << stream.device() << std::endl;
        initialize();
        std::cout << "Current device: " << stream.device() << std::endl;
    }

    // default destructor
    ~TracccGpuStandalone() = default;

    void initialize();
    void run(std::vector<traccc::io::csv::cell> cells);
    std::vector<traccc::io::csv::cell> read_from_array(const std::vector<std::vector<double>> &data);
};

void TracccGpuStandalone::initialize()
{
    // HACK: hard code location of detector and digitization file
    detector_opts.detector_file = "/global/cfs/projectdirs/m3443/data/traccc-aaS/data/tml_detector/trackml-detector.csv";
    detector_opts.digitization_file = "/global/cfs/projectdirs/m3443/data/traccc-aaS/data/tml_detector/default-geometric-config-generic.json";

    // read in geometry
    auto geom_data = traccc::io::read_geometry(detector_opts.detector_file,
                                            (detector_opts.use_detray_detector ? traccc::data_format::json : traccc::data_format::csv));
    surface_transforms = std::move(geom_data.first);
    barcode_map = std::move(geom_data.second);


    // cfg.add_file(detector_opts.detector_file);

    // Read the digitization configuration file
    digi_cfg = std::make_unique<traccc::digitization_config>(traccc::io::read_digitization_config(detector_opts.digitization_file));

    return;
}

void TracccGpuStandalone::run(std::vector<traccc::io::csv::cell> cells)
{
    traccc::io::cell_reader_output read_out(mr.host);

    // Read the cells from the relevant event file into host memory.
    read_cells(read_out, cells, &surface_transforms, digi_cfg.get(), barcode_map.get(), true);

    const traccc::cell_collection_types::host& cells_per_event =
        read_out.cells;
    const traccc::cell_module_collection_types::host&
        modules_per_event = read_out.modules;

    // Create device copy of input collections
    traccc::cell_collection_types::buffer cells_buffer(
        cells_per_event.size(), mr.main);
    copy(vecmem::get_data(cells_per_event), cells_buffer);
    traccc::cell_module_collection_types::buffer modules_buffer(
        modules_per_event.size(), mr.main);
    copy(vecmem::get_data(modules_per_event), modules_buffer);

    // Reconstruct it into spacepoints on the device.
    traccc::measurement_collection_types::buffer measurements_cuda_buffer(
            0, *mr.host);
    measurements_cuda_buffer = ca_cuda(cells_buffer, modules_buffer);
    ms_cuda(measurements_cuda_buffer);
    
    stream.synchronize();

    copy(measurements_cuda_buffer, measurements_per_event_cuda)->wait();

    stream.synchronize();

    // Print out measurements!
    std::cout << measurements_per_event_cuda.size() << std::endl;

    for (std::size_t i = 0; i < 10; ++i) {
        auto measurement = measurements_per_event_cuda.at(i);
        std::cout << "Measurement ID: " << measurement.measurement_id << std::endl;
        std::cout << "Local coordinates: [" << measurement.local[0] << ", " << measurement.local[1] << "]" << std::endl; 
    }

    return;
}

// deal with input data

std::vector<traccc::io::csv::cell> read_csv(const std::string &filename)
{
    std::vector<traccc::io::csv::cell> cells;
    auto reader = traccc::io::csv::make_cell_reader(filename);
    traccc::io::csv::cell iocell;

    while (reader.read(iocell))
    {
        cells.push_back(iocell);
    }

    return cells;
}

std::map<std::uint64_t, std::map<traccc::cell, float, cell_order>> fill_cell_map(const std::vector<traccc::io::csv::cell> &cells, 
                                                                                    unsigned int &nduplicates)
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
                const std::map<std::uint64_t, 
                detray::geometry::barcode> *barcode_map, 
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
