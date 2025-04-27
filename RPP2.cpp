#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <mpi.h>

// Apply custom median filter to an image
void applyMedianFilter(const cv::Mat& input, cv::Mat& output, int windowSize) {
    int halfWindow = windowSize / 2;

    // Process each channel separately
    for (int c = 0; c < input.channels(); c++) {
        for (int y = 0; y < input.rows; y++) {
            std::vector<uchar> window;
            window.reserve(windowSize * windowSize);

            for (int x = 0; x < input.cols; x++) {
                window.clear();

                // Gather pixel values from the neighborhood
                for (int wy = -halfWindow; wy <= halfWindow; wy++) {
                    for (int wx = -halfWindow; wx <= halfWindow; wx++) {
                        // Calculate the position with boundary checking
                        int nx = std::min(std::max(x + wx, 0), input.cols - 1);
                        int ny = std::min(std::max(y + wy, 0), input.rows - 1);

                        // Add the pixel value to the window
                        if (input.channels() == 1) {
                            window.push_back(input.at<uchar>(ny, nx));
                        }
                        else {
                            cv::Vec3b pixel = input.at<cv::Vec3b>(ny, nx);
                            window.push_back(pixel[c]);
                        }
                    }
                }

                // Sort the window values and pick the median
                std::nth_element(window.begin(), window.begin() + window.size() / 2, window.end());
                uchar medianValue = window[window.size() / 2];

                // Set the output pixel value
                if (output.channels() == 1) {
                    output.at<uchar>(y, x) = medianValue;
                }
                else {
                    cv::Vec3b& pixel = output.at<cv::Vec3b>(y, x);
                    pixel[c] = medianValue;
                }
            }
        }
    }
}

// Process a single image with median filter
void processImage(const std::string& inputFile, const std::string& outputFile, int windowSize) {
    // Load image using OpenCV (supports various formats including JPG)
    cv::Mat image = cv::imread(inputFile);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << inputFile << std::endl;
        return;
    }

    // Create output image with same size and type
    cv::Mat result = image.clone();

    // Apply our custom median filter
    applyMedianFilter(image, result, windowSize);

    // Save the processed image
    cv::imwrite(outputFile, result);
}

// Find all images in a directory
std::vector<std::string> findImageFiles(const std::string& directory, const std::vector<std::string>& extensions) {
    std::vector<std::string> imageFiles;

    try {
        // Check if directory exists
        if (!std::filesystem::exists(directory) || !std::filesystem::is_directory(directory)) {
            std::cerr << "Invalid directory: " << directory << std::endl;
            return imageFiles;
        }

        // Iterate through directory entries
        for (const auto& entry : std::filesystem::directory_iterator(directory)) {
            if (entry.is_regular_file()) {
                std::string path = entry.path().string();
                std::string ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

                // Check if file has one of the specified extensions
                for (const auto& validExt : extensions) {
                    if (ext == validExt) {
                        imageFiles.push_back(path);
                        break;
                    }
                }
            }
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return imageFiles;
}

// Process images in parallel using MPI
void processImagesInParallel(const std::vector<std::string>& inputFiles,
    const std::string& outputDir,
    const std::string& outputPrefix,
    int windowSize) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Only create the output directory on rank 0
    if (rank == 0) {
        try {
            if (!std::filesystem::exists(outputDir)) {
                std::filesystem::create_directories(outputDir);
            }
        }
        catch (const std::exception& e) {
            std::cerr << "Error creating output directory: " << e.what() << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
            return;
        }
    }

    // Make sure all processes wait until the directory is created
    MPI_Barrier(MPI_COMM_WORLD);

    // Get total number of images
    int totalImages = inputFiles.size();

    // Calculate how many images each process should handle
    int imagesPerProcess = totalImages / size;
    int remainingImages = totalImages % size;

    // Determine the start and end indices for this process
    int startIdx = rank * imagesPerProcess + std::min(rank, remainingImages);
    int endIdx = startIdx + imagesPerProcess + (rank < remainingImages ? 1 : 0);

    // Log the assignment (from master only to avoid clutter)
    if (rank == 0) {
        std::cout << "Distributing " << totalImages << " images among " << size << " processes:" << std::endl;
        for (int i = 0; i < size; i++) {
            int start = i * imagesPerProcess + std::min(i, remainingImages);
            int end = start + imagesPerProcess + (i < remainingImages ? 1 : 0);
            std::cout << "  Process " << i << " will handle " << (end - start) << " images" << std::endl;
        }
    }

    // Process assigned images
    for (int i = startIdx; i < endIdx; i++) {
        std::string inputFile = inputFiles[i];
        std::filesystem::path inputPath(inputFile);
        std::string filename = inputPath.filename().string();
        std::string outputFile = outputDir + "/" + outputPrefix + filename;

        std::cout << "Process " << rank << " processing image " << (i + 1)
            << "/" << totalImages << ": " << filename << std::endl;

        // Process the image
        processImage(inputFile, outputFile, windowSize);

        std::cout << "Process " << rank << " completed: " << filename << std::endl;
    }

    // Wait for all processes to finish
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "All images processed successfully." << std::endl;
    }
}

int main(int argc, char* argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Default parameters
    int windowSize = 3;                   // Size of the median filter window (3x3)
    std::string inputDir = "./";          // Input directory
    std::string outputDir = "./filtered"; // Output directory
    std::string outputPrefix = "";        // Output filename prefix

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--window") == 0 && i + 1 < argc) {
            windowSize = std::atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            inputDir = argv[++i];
        }
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            outputDir = argv[++i];
        }
        else if (strcmp(argv[i], "--prefix") == 0 && i + 1 < argc) {
            outputPrefix = argv[++i];
        }
        else if (rank == 0) {
            // Usage instructions (only print from rank 0)
            std::cout << "Usage: " << argv[0] << " [options]\n"
                << "Options:\n"
                << "  --window N     Set median filter window size (default: 3)\n"
                << "  --input DIR    Set input directory (default: ./)\n"
                << "  --output DIR   Set output directory (default: ./filtered)\n"
                << "  --prefix P     Set output filename prefix (default: none)\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }

    // Supported file extensions
    std::vector<std::string> supportedExtensions = { ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff" };

    // Find all image files in the input directory (only master process)
    std::vector<std::string> inputFiles;
    if (rank == 0) {
        std::cout << "Scanning directory: " << inputDir << " for image files..." << std::endl;
        inputFiles = findImageFiles(inputDir, supportedExtensions);

        if (inputFiles.empty()) {
            std::cout << "No image files found in directory: " << inputDir << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }

        std::cout << "Found " << inputFiles.size() << " image files to process." << std::endl;
    }

    // Broadcast the number of files to all processes
    int numFiles = (rank == 0) ? static_cast<int>(inputFiles.size()) : 0;
    MPI_Bcast(&numFiles, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Each process needs the full list of files to determine its portion
    if (rank != 0) {
        inputFiles.resize(numFiles);
    }

    // Broadcast each filename to all processes
    for (int i = 0; i < numFiles; i++) {
        int filenameSize = 0;
        if (rank == 0) {
            filenameSize = static_cast<int>(inputFiles[i].size());
        }
        MPI_Bcast(&filenameSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

        char* buffer = new char[filenameSize + 1];
        if (rank == 0) {
            strcpy_s(buffer, filenameSize + 1, inputFiles[i].c_str());
        }

        MPI_Bcast(buffer, filenameSize, MPI_CHAR, 0, MPI_COMM_WORLD);

        if (rank != 0) {
            buffer[filenameSize] = '\0';
            inputFiles[i] = std::string(buffer);
        }

        delete[] buffer;
    }

    // Measure execution time
    double startTime = MPI_Wtime();

    // Process all images in parallel
    processImagesInParallel(inputFiles, outputDir, outputPrefix, windowSize);

    // Calculate and print execution time
    double endTime = MPI_Wtime();
    if (rank == 0) {
        std::cout << "Total processing time: " << (endTime - startTime) << " seconds" << std::endl;
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}