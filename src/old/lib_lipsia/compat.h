/*
**
** compatibility wrappers
**
** E. Reimer, Jan 2014
*/

#ifndef COMPAT_H_INCLUDED
#define COMPAT_H_INCLUDED

#ifdef __cplusplus
extern "C" {
#endif

int lipsia_gettime(struct timespec* t);

#ifdef __cplusplus
}
#endif
#endif