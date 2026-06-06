#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>

/* Windows-compatible version of c_ldpc.c
   Replace all `long` with `int64_t` so pointer sizes match
   Python ctypes on 64-bit Windows where sizeof(long)==4 but sizeof(int64_t)==8.
   Algorithm is identical — only the integer type declarations change.
   Jossy, September 2018 / Win64 patch June 2026
*/

#define MAX_ITCOUNT 200

double Lxor(double L1, double L2, int corr_flag);
double Lxfb(double *L, int64_t dc, int corr_flag);

int sumprod(double *ch, int64_t *vdeg, int64_t *cdeg, int64_t *intrlv,
            int Nv, int Nc, int Nmsg, double *app)
{
  double *msg;
  int j, k, imsg, stopflag, itcount;
  double aggr;

  msg = calloc(Nmsg, sizeof(double));
  if (msg == NULL) return(-1);

  for (itcount = 0 ; itcount < MAX_ITCOUNT ; itcount++) {

    for (j = 0, imsg = 0 ; j < Nv ; j++) {
      for (k = 0, aggr = ch[j] ; k < vdeg[j] ; k++, imsg++)
        aggr += msg[intrlv[imsg]];
      imsg -= vdeg[j];
      for (k = 0; k < vdeg[j] ; k++, imsg++)
        msg[intrlv[imsg]] = aggr - msg[intrlv[imsg]];
      app[j] = aggr;
    }

    stopflag = 0;

    for (j = 0, imsg = 0 ; j < Nc ; j++) {
      for (k = 0, aggr = 1.0 ; k < cdeg[j] ; k++, imsg++)
        aggr *= (msg[imsg] = tanh(msg[imsg]/2.0));
      if (stopflag == 0 && 2.0 * atanh(aggr) <= 0.0)
        stopflag = 1;
      imsg -= cdeg[j];
      for (k = 0; k < cdeg[j] ; k++, imsg++)
        msg[imsg] = 2.0 * atanh(aggr / msg[imsg]);
    }

    if (!stopflag) break;
  }

  free(msg);
  return(itcount);
}

int sumprod2(double *ch, int64_t *vdeg, int64_t *cdeg, int64_t *intrlv,
             int Nv, int Nc, int Nmsg, double *app)
{
  double *msg;
  int j, k, imsg, stopflag, itcount;
  double aggr;

  msg = calloc(Nmsg, sizeof(double));
  if (msg == NULL) return(-1);

  for (itcount = 0 ; itcount < MAX_ITCOUNT ; itcount++) {

    for (j = 0, imsg = 0 ; j < Nv ; j++) {
      for (k = 0, aggr = ch[j] ; k < vdeg[j] ; k++, imsg++)
        aggr += msg[intrlv[imsg]];
      imsg -= vdeg[j];
      for (k = 0; k < vdeg[j] ; k++, imsg++)
        msg[intrlv[imsg]] = aggr - msg[intrlv[imsg]];
      app[j] = aggr;
    }

    stopflag = 0;

    for (j = 0, imsg = 0 ; j < Nc ; j++) {
      aggr = Lxfb(&(msg[imsg]), cdeg[j], 1);
      if (stopflag == 0 && aggr <= 0.0)
        stopflag = 1;
      imsg += cdeg[j];
    }

    if (!stopflag) break;
  }

  free(msg);
  return(itcount);
}

double Lxor(double L1, double L2, int corr_flag)
{
  double L;
  if (signbit(L1) == signbit(L2)) L = 1.0; else L = -1.0;
  L *= fmin(fabs(L1), fabs(L2));
  if (corr_flag) {
    L += log(1+exp(-fabs(L1+L2)));
    L -= log(1+exp(-fabs(L1-L2)));
  }
  return(L);
}

#define MAXDC 25

double Lxfb(double *L, int64_t dc, int corr_flag)
{
  double f[MAXDC];
  double b[MAXDC];
  int k;
  for (k = 1, f[0] = L[0], b[dc-1] = L[dc-1] ; k < dc ; k++) {
    f[k]      = Lxor(f[k-1],    L[k],      corr_flag);
    b[dc-k-1] = Lxor(b[dc-k],  L[dc-k-1], corr_flag);
  }
  for (k = 1, L[0] = b[1], L[dc-1] = f[dc-2] ; k < dc-1 ; k++)
    L[k] = Lxor(f[k-1], b[k+1], corr_flag);
  return(b[0]);
}

int minsum(double *ch, int64_t *vdeg, int64_t *cdeg, int64_t *intrlv,
           int Nv, int Nc, int Nmsg, double *app, double correction_factor)
{
  double *msg;
  int j, k, imsg, stopflag, itcount;
  double aggr;

  msg = calloc(Nmsg, sizeof(double));
  if (msg == NULL) return(-1);

  for (itcount = 0 ; itcount < MAX_ITCOUNT ; itcount++) {
    for (j = 0, imsg = 0 ; j < Nv ; j++) {
      for (k = 0, aggr = ch[j] ; k < vdeg[j] ; k++, imsg++)
        aggr += msg[intrlv[imsg]];
      imsg -= vdeg[j];
      for (k = 0; k < vdeg[j] ; k++, imsg++)
        msg[intrlv[imsg]] = aggr - msg[intrlv[imsg]];
      app[j] = aggr;
    }

    stopflag = 0;

    for (j = 0, imsg = 0 ; j < Nc ; j++, imsg += cdeg[j]) {
      aggr = Lxfb(&(msg[imsg]), cdeg[j], 0);
      if (stopflag == 0 && aggr <= 0.0)
        stopflag = 1;
      for (k = 0 ; k < cdeg[j]; k++)
        msg[imsg+k] *= correction_factor;
    }

    if (!stopflag) break;
  }

  free(msg);
  return(itcount);
}
