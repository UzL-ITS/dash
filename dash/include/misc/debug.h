#ifndef DEBUG_H
#define DEBUG_H

#ifdef NDEBUG
#define DEBUG(x)
#else
#define DEBUG(x) do { std::cerr << x << std::endl; } while (0);
#endif

#endif