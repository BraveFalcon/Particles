#pragma once

#include <fstream>

class LogStream : std::ostream {
private:
    std::ofstream file;
public:
    void open(const std::string &log_file_path) {
        file.close();
        file.open(log_file_path);
        if (!file) {
            std::cerr << "Can't open log file" << std::endl;
            exit(1);
        }
    }

    void close() { file.close(); }

    LogStream() = default;

    explicit LogStream(const std::string &log_file_path) : file(log_file_path) {
        if (!file) {
            std::cerr << "Can't open log file" << std::endl;
            exit(1);
        }
    }

    template<typename T>
    LogStream &operator<<(T val) {
        file << val;
        std::cout << val;
        return *this;
    }

    void flush() {
        std::cout.flush();
        file.flush();
    }


    ~LogStream() { close(); }
};
