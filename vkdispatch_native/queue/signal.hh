#ifndef _SIGNAL_SRC_SIGNAL_H
#define _SIGNAL_SRC_SIGNAL_H

#include "../base.hh"

#include <atomic>
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
     * @brief Creates a new signal.
     */
    Signal(struct Context* context);

    /**
     * @brief Notifies the signal.
     *
     * This function sets the state of the signal to true, indicating that the condition has occurred.
     * It wakes up any waiting threads.
     */
    void notify(int queue_index, uint64_t timestamp);

    /**
     * @brief Resets the signal to the initial state.
     *
     * This function sets the state of the signal to false, indicating that the condition has not occurred.
     * It allows threads to wait for the signal again.
     */
    void reset();

    /**
     * @brief Waits for the signal.
     *
     * This function blocks the calling thread until the signal is notified.
     * If the signal is already in the notified state, the function returns immediately.
     * 
     * This function will return after one second even if the signal is not notified, to prevent deadlocks.
     * @return true if the signal was notified, false if the wait timed out.
     */
    bool try_wait(bool wait_for_timestamp, int queue_index);

private:
    bool try_host_wait();
    bool try_device_wait(int queue_index);

public:

    struct Context* ctx;
    uint64_t timestamp;
    int timestamp_queue_index;
    std::mutex mutex;
    std::condition_variable cv;
    std::atomic<bool> state;
};


#endif // _SIGNAL_SRC_SIGNAL_H