/*! \file
  3D connected component labelling

Each foreground voxel receives a label indicating
its membership in a connected component.
The algorithm is based on depth first search.

\par Reference:
G. Lohmann (1998). "Volumetric Image Analysis",
John Wiley & Sons, Chichester, England.

\par Author:
Gabriele Lohmann, MPI-CBS
*/

/* From the Vista library: */
#include <viaio/Vlib.h>
#include <viaio/mu.h>
#include <via/via.h>


/* From the standard C libaray: */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


typedef struct {
  Voxel *A;
  int front,rear;
} Queue;

static Queue queue;
static int QueueSize=0;
static int msize=0;
typedef int BOOLEAN;

static void QueueClear(Queue *);
static BOOLEAN QueueEmpty(Queue *); 
static BOOLEAN deQueue(Queue *, Voxel *);
static BOOLEAN enQueue(Queue *, Voxel);

#define VPutPixel(image,b,r,c,repn,value) \
  if (repn == VShortRepn) VPixel((image),(b),(r),(c),VShort) = (value); \
  else VPixel((image),(b),(r),(c),VUByte) = (value);


/*!
\fn VImage VLabelImage3d(VImage src, VImage dest, int neighb, VRepnKind repn, int *numlabels)
\param src  input image (bit repn)
\param dest output image (ubyte or short repn)
\param neighb adjacency type (6 or 26)
\param repn pixel repn of the output image (VUByteRepn or VShortRepn). If
VUByteRepn is selected, then no more than 255 connected components can be 
identified.
\param numlabels ptr to the number of labels found.
*/
VImage
VLabelImage3d(VImage src,VImage dest,int neighb,VRepnKind repn,int *numlabels)
{
  int i,nbands,nrows,ncols,npixels;
  VBit *src_pp;
  Voxel v,vv;
  int label,n,nblack;
  int b0,b1,r0,r1,c0,c1,b,r,c,bb,rr,cc;
  int ba[6],ra[6],ca[6],m;
  
  msize = 0;
  if (VPixelRepn(src) != VBitRepn) 
    VError("Input image must be of type VBit");

  nbands  = VImageNBands(src);
  nrows   = VImageNRows(src);
  ncols   = VImageNColumns(src);
  npixels = nbands * nrows * ncols;

  switch(repn) {
  case VUByteRepn:
    dest = VSelectDestImage("VLabel3d",dest,nbands,nrows,ncols,VUByteRepn);
    if (! dest) return NULL;
    VFillImage(dest,VAllBands,0);
    break;

  case VShortRepn:
    dest = VSelectDestImage("VLabel3d",dest,nbands,nrows,ncols,VShortRepn);
    if (! dest) return NULL;
    VFillImage(dest,VAllBands,0);
    break;

  default:
    VError("Output image representation must be either ubyte or short.");
  }

  /*
  ** set up Queue
  */
  if (numlabels != NULL) *numlabels = 0;

  nblack = 0;
  src_pp = (VBit *) VImageData(src);
  for (i=0; i<npixels; i++) 
    if (*src_pp++ > 0) nblack++;
  if (nblack < 1) return dest;

  QueueSize = (float)(nblack) * 0.666;
  if (QueueSize < 128) QueueSize=128;
  queue.A = (Voxel *) VMalloc(sizeof(Voxel) * QueueSize);
  msize = 0;
  label = 0;

  /*
  ** depth first search
  */
  switch(neighb) {

    /*
    ** 26 - adjacency
    */
  case 26:

    for (b=0; b<nbands; b++) {
      for (r=0; r<nrows; r++) {
	for (c=0; c<ncols; c++) {
	  if (VPixel(src,b,r,c,VBit) == 0) continue;

	  if (repn == VUByteRepn 
	      && VPixel(dest,b,r,c,VUByte) > 0) continue;
	  if (repn == VShortRepn 
	      && VPixel(dest,b,r,c,VShort) > 0) continue;

	  QueueClear(&queue);

	  label++;
	  if (label >= VPixelMaxValue (dest)) {
	    VWarning("Number of labels exceeds maximum (%d)",label);
	    return dest;
	  }
	  n = 1;

	  v.b = b;
	  v.r = r;
	  v.c = c;
	  if (enQueue(&queue,v) == FALSE) VError(" error in enQueue");
	  VPutPixel(dest,b,r,c,repn,label);

	  while (! QueueEmpty(&queue)) {

	    deQueue(&queue,&v);

	    b0 = (v.b > 0) ? v.b - 1 : 0;
	    b1 = (v.b + 1 < nbands) ? v.b + 1 : nbands - 1;
	    for (bb=b0; bb<=b1; bb++) {

	      r0 = (v.r > 0) ? v.r - 1 : 0;
	      r1 = (v.r + 1 < nrows) ? v.r + 1 : nrows - 1;
	      for (rr=r0; rr<=r1; rr++) {

		c0 = (v.c > 0) ? v.c - 1 : 0;
		c1 = (v.c + 1 < ncols) ? v.c + 1 : ncols - 1;
		for (cc=c0; cc<=c1; cc++) {

		  if (bb == v.b && rr == v.r && cc == v.c) continue;
		  if (VPixel(src,bb,rr,cc,VBit) == 0) continue;
		  if (repn == VUByteRepn
		      && VPixel(dest,bb,rr,cc,VUByte) > 0) continue;
		  if (repn == VShortRepn 
		      && VPixel(dest,bb,rr,cc,VShort) > 0) continue;
		  vv.b = bb;
		  vv.r = rr;
		  vv.c = cc;
		  if (enQueue(&queue,vv) == FALSE) VError(" error in enQueue");
		  VPutPixel(dest,bb,rr,cc,repn,label);
		  n++;
		}
	      }
	    }
	  }
	}
      }
    }
    break;

    /*
    ** 6 -adjacency
    */
  case 6:

    for (b=0; b<nbands; b++) {
      for (r=0; r<nrows; r++) {
	for (c=0; c<ncols; c++) {
	  if (VPixel(src,b,r,c,VBit) == 0) continue;

	  if (repn == VUByteRepn 
	      && VPixel(dest,b,r,c,VUByte) > 0) continue;
	  if (repn == VShortRepn 
	      && VPixel(dest,b,r,c,VShort) > 0) continue;

	  QueueClear(&queue);

	  label++;
	  if (label >= VPixelMaxValue (dest)) {
	    VWarning("Number of labels exceeds maximum (%d)",label);
	    VFree(queue.A);
	    return dest;
	  }
	  n = 1;

	  v.b = b;
	  v.r = r;
	  v.c = c;
	  if (enQueue(&queue,v) == FALSE) VError(" error in enQueue");
	  VPutPixel(dest,b,r,c,repn,label);

	  while (! QueueEmpty(&queue)) {

	    deQueue(&queue,&v);

	    for (i=0; i<6; i++) {
	      ba[i] = v.b;
	      ra[i] = v.r;
	      ca[i] = v.c;
	    }
	    m = 0;
	    if (v.b > 0) ba[m++] = v.b - 1;
	    if (v.r > 0) ra[m++] = v.r - 1;
	    if (v.c > 0) ca[m++] = v.c - 1;

	    if (v.b < nbands - 1) ba[m++] = v.b + 1;
	    if (v.r < nrows - 1)  ra[m++] = v.r + 1;
	    if (v.c < ncols - 1)  ca[m++] = v.c + 1;

	    for (i=0; i<m; i++) {
	      if (VPixel(src,ba[i],ra[i],ca[i],VBit) == 0) continue;
	      if (repn == VUByteRepn 
		  && VPixel(dest,ba[i],ra[i],ca[i],VUByte) > 0) continue;
	      if (repn == VShortRepn 
		  && VPixel(dest,ba[i],ra[i],ca[i],VShort) > 0) continue;

	      VPutPixel(dest,ba[i],ra[i],ca[i],repn,label);
	      vv.b = ba[i];
	      vv.r = ra[i];
	      vv.c = ca[i];
	      if (enQueue(&queue,vv) == FALSE) VError(" error in enQueue");
	      n++;
	    }
	  }
	}
      }
    }
    break;

  default:
    ;
  }

  VFree(queue.A);
  if (numlabels != NULL) *numlabels = label;
  VCopyImageAttrs (src, dest);
  return dest;
}


/*
** array implementation of queues, following
** Aho, Ullman, "Foundations of Computer Science in C"
** Computer Science Press, 1995, pp. 319 ff
**
*/

#define TRUE  1
#define FALSE 0

static void QueueClear(Queue *pQ)
{
  pQ->front = 0;
  pQ->rear  = 0;
}

static BOOLEAN
QueueEmpty(Queue *pQ) 
{
  return (pQ->front == pQ->rear);
}


static BOOLEAN
deQueue(Queue *pQ, Voxel *pe)
{
  if (pQ->front == pQ->rear) {
    fprintf(stderr," deQueue: empty\n");
    return FALSE;
  }
  else {
    (*pe) = pQ->A[(pQ->rear)++];
    return TRUE;
  }
}

static BOOLEAN 
enQueue(Queue *pQ, Voxel e)
{
  if (pQ->front > msize) msize = pQ->front;
  
  if (pQ->front < QueueSize - 1) {
    pQ->A[(pQ->front)++] = e;
    return TRUE;
  }
  else if (pQ->rear > 2) {
    pQ->A[--(pQ->rear)] = e;
    return TRUE;
  }
  else {
    QueueSize += QueueSize * 0.333;
    queue.A = (Voxel *) VRealloc(queue.A,sizeof(Voxel) * QueueSize);
    pQ->A[(pQ->front)++] = e;
    return TRUE;
  }
}

