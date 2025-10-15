#pragma  once
#include <iostream>
#include <chrono>
#include <cmath>
#include <string>

namespace {
    class Timer {
        std::chrono::_V2::system_clock::time_point startTime;
        std::chrono::_V2::system_clock::time_point endTime;
        static int counter;
        std::string TimerName;

    public:
        Timer(std::string name = "None") {
            TimerName = name; 
            startTime = std::chrono::high_resolution_clock::now();
            counter++;
        }
        ~Timer() {
            endTime = std::chrono::high_resolution_clock::now();
            auto differenz = endTime-startTime;
            counter--;

            auto minutes = std::chrono::duration_cast< std::chrono::minutes >( differenz );
            differenz -= minutes;

            auto seconds = std::chrono::duration_cast< std::chrono::seconds >( differenz );
            differenz -= seconds;

            auto milliseconds = std::chrono::duration_cast< std::chrono::milliseconds >( differenz );
            differenz -= milliseconds;

            auto microseconds = std::chrono::duration_cast< std::chrono::microseconds >( differenz );
            differenz -= microseconds;

            auto nanoseconds = std::chrono::duration_cast< std::chrono::nanoseconds >( differenz );
            if (TimerName != "None") {
                std::cout << "Timer " << TimerName << " finished in: ";
            } else {
                std::cout << "Timer " << counter << " finished in: ";
            }
            std::cout << minutes.count() << " Minutes, "
                        << seconds.count() << " Seconds, "
                        << milliseconds.count() << " Milliseconds, "
                        << microseconds.count() << " Microseconds, "
                        << nanoseconds.count() << " Nanoseconds." << std::endl;

        }
    };   int Timer::counter = 1;
}





