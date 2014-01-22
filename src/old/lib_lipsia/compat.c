/*
**
** compatibility wrappers
**
** E. Reimer, Jan 2014
*/

#include <time.h>

int lipsia_gettime(struct timespec* t) {
#ifdef __MACH__
    struct timeval now;
    int rv = gettimeofday(&now, NULL);
    if (rv) return rv;
    t->tv_sec  = now.tv_sec;
    t->tv_nsec = now.tv_usec * 1000;
    return 0;
#else
	return lipsia_gettime(t);
#endif
}
