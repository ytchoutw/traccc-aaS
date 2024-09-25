#include "TracccGpuStandalone.hpp"
#include <chrono>


int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        std::cout << "Not enough arguments, minimum requirement two of the form: " << std::endl;
        std::cout << argv[0] << " <event_file> " << "<deviceID>" << std::endl;
        return -1;
    }

    std::string event_file = std::string(argv[1]);
    std::cout << "Running " << argv[0] << " on " << event_file << std::endl;

    int deviceID = std::stoi(argv[2]);
    auto cells = read_csv(event_file);
    TracccGpuStandalone standalone(cells, deviceID);

    std::vector<double> timeProcessOneEvent;

    // warm up
    for (int i = 0; i < 10; i++)
    {
        standalone.run();
    }

    for (int i = 0; i < 100; i++)
    {
        auto start = std::chrono::high_resolution_clock::now();
        standalone.run();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        timeProcessOneEvent.push_back(duration.count());
    }

    std::cout << " " << std::endl;
    std::cout << "Estimated performance of standalone: " << std::endl;
    std::cout << "Average time to process one event: " << std::accumulate(timeProcessOneEvent.begin(), 
        timeProcessOneEvent.end(), 0.0) / timeProcessOneEvent.size() << " s" << std::endl;
    std::cout << "Throughput: " << timeProcessOneEvent.size() / std::accumulate(timeProcessOneEvent.begin(), 
        timeProcessOneEvent.end(), 0.0) << " events/s" << std::endl;

    return 0;
}