#pragma once

#include <thread>
#include <vector>


class SliceMultiThread
{
public:
	template<typename T, typename Functor>
	static void multiThread(T& obj, unsigned threadNum, Functor fun);

	template<typename T, typename Functor>
	static void multiThread(const T& obj, unsigned threadNum, Functor fun);
};

template<typename T, typename Functor>
inline void SliceMultiThread::multiThread(T &obj, unsigned threadNum, Functor fun)
{
    std::vector<std::thread> threads;

    for (decltype(threadNum) t = 0; t < threadNum; ++t) {
        threads.emplace_back([&, t] {
            auto totalSeq = obj.size();
            for (auto i = static_cast<decltype(totalSeq)>(t); i < totalSeq; i += threadNum) {
                fun(t, i, obj[i]);
            }
            });
    }

    for (auto &thread : threads) {
        thread.join();
    }
}

template<typename T, typename Functor>
void SliceMultiThread::multiThread(const T &obj, unsigned threadNum, Functor fun)
{
    std::vector<std::thread> threads;

    for (decltype(threadNum) t = 0; t < threadNum; ++t) {
        threads.emplace_back([&, t] {
            auto totalSeq = obj.size();
            for (auto i = static_cast<decltype(totalSeq)>(t); i < totalSeq; i += threadNum) {
                fun(t, i, obj[i]);
            }
            });
    }

    for (auto &thread : threads) {
        thread.join();
    }
}

