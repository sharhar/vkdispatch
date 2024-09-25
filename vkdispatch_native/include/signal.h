#ifndef _SIGNAL_SRC_SIGNAL_H
#define _SIGNAL_SRC_SIGNAL_H

#include <mutex>
#include <condition_variable>

/**
 * @brief Represents a signal that can be used for synchronization.
 *
 * This class provides a simple signal mechanism that can be used for synchronization between threads.
 * It allows one thread to notify other threads that a certain condition has occurred.
 */
class Signal {
public:
    /**
     * @brief Creates a new signal. Must be called from the main thread!!
     */
    Signal();

    /**
     * @brief Notifies the signal. Must be called from a stream thread!!
     *
     * This function sets the state of the signal to true, indicating that the condition has occurred.
     * It wakes up any waiting threads.
     */
    void notify();

    /**
     * @brief Waits for the signal. Must be called from the main thread!!
     *
     * This function blocks the calling thread until the signal is notified.
     * If the signal is already in the notified state, the function returns immediately.
     */
    void wait();
    
    std::mutex mutex;
    std::condition_variable cv;
    bool state;
};


#endif // _SIGNAL_SRC_SIGNAL_H