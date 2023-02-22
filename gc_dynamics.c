/*******************************************************************************
 *
 *  FILE:         gc_dynamics.c
 *
 *  DATE:         17 November 2019
 *
 *  AUTHORS:      Louis Kang, University of California, Berkeley
 *
 *                Based on code by Yoram Burak and Ila Fiete
 *                Based on code by Louis Kang and Vijay Balasubramanian
 *
 *  LICENSING:    CC BY 4.0
 *
 *  REFERENCE:    Kang L, DeWeese MR. Replay as wavefronts and theta sequences
 *                as bump oscillations in a grid cell attractor network. eLife
 *                8, e46351 (2019). doi:10.7554/eLife.46351
 *
 *  PURPOSE:      Simulation of a continuous attractor grid network with
 *                leaky integrate-and-fire dynamics and time-varying inputs.
 *
 *  DEPENDENCIES: FFTW3 single-precision (http://www.fftw.org/)
 *                ziggurat random number generation package by John Burkardt
 *           (https://people.sc.fsu.edu/~jburkardt/c_src/ziggurat/ziggurat.html)
 *
 *******************************************************************************
 *
 *
 *  This is the source code for simulating a continuous attractor grid network
 *  with leaky integrate-and-fire dynamics. There are four excitatory
 *  populations and one inhibitory population. Inputs to the populations can
 *  vary with time to simulate animal motion and changes in behavioral state.
 *  Simulation setup involves a few "flow" phases during which the grid activity
 *  patterns are self-organized and refined. Then the main simulation begins
 *  during which spikes and/or membrane potentials are output. See the reference
 *  for full details of the simulation and interpretation of its results.
 *
 *  Sample usage for the compiled executable file gc_dynamics, trajectory file
 *  traj_1500-2000-01.txt, and landmark learning trajectory file
 *  traj-learn_1500.txt that generates simulations presented in Figures 2, 3, 5,
 *  and 7 of the reference:
 *    gc_dynamics -fileroot /path/to/output \
 *      -laps_sim 36 -laps_flow 4 \
 *      -w_in 400. -w_etoi 8. -w_etoe 8. \
 *      -l_in 12. -l_ex 3. -w_shift 3. \
 *      -a_in 0.72 -a_th 0.20 \
 *      -a_in_rmag 0.0 \
 *      -a_ex_mag 2.0 -a_ex_min 0.8 -a_ex_shoul 1.2 \
 *      -a_ex_rmag 1.6 -a_ex_rmin 0.8 -a_ex_rshoul 0.9 \
 *      -vgain 2.5 \
 *      -traj traj_1500-2000-01.txt \
 *      -tlearn 500 -tlm_flow 500 -lm_cutoff 0.05 \
 *      -traj_lm traj-learn_1500.txt \
 *      -no_psi
 *
 *  This command will output the following in the directory /path/to/ (the
 *  total size will be about 4.3 GB):
 *    - output_a-*.txt:         various input profiles
 *    - output_params.txt:      simulation parameter values
 *    - output_psi-setup*.txt:  membrane potentials at various points during
 *                              simulation setup
 *    - output_s-setup*.txt:    spikes at various points during simulation setup
 *    - output_s*-lap*.dat      binary files containing spikes from various
 *                              populations during main simulation
 *    - output_traj.txt:        animal trajectory
 *    - output_w-*.txt:         various recurrent connection profiles
 *
 ******************************************************************************/

#include <complex.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <libgen.h>

#include "fftw3.h"
#include "ziggurat.h"



//==============================================================================
// Parameters and default value declarations
//==============================================================================


#define PI 3.141592654
#define UNDEF -999


// Simulation ------------------------------------------------------------------
char fileroot[256] = "";    // prefix for output files
uint32_t randseed = 0;      // 0: use time as random seed
float r4fn[128];            // for ziggurat package
uint32_t r4kn[128];         // for ziggurat package
float r4wn[128];            // for ziggurat package

// Iterations ------------------------------------------------------------------
int laps_sim = 8;         // number of main simulation laps
int laps_flow = 4;        // number of trajectory flow laps
int tflow1 = 500;         // flow population activity with no velocity
int tflow2 = 8000;        // flow population activity with constant velocity
int trun = UNDEF;         // length of run per lap
int treplay = UNDEF;      // length of replay per lap
int tlap;                 // tlap = trun + treplay
int tscreen = 200;        // print to screen every tscreen iterations

// Network --------------------------------------------------------------------
int periodic = 0;                   // periodic boundary conditions
int n = UNDEF;                      // number of neurons per side
int np = 256;                       // size of fft arrays with padding
int pad;                            // size of padding on each side
int n2, n2bin, np2, npfft;          // for convenience
float dt = 1.;                      // time step in ms
float tau_in = 20.;                 // inhibitory neuron time constant in ms
float tau_ex = 40.;                 // excitatory neuron time constant in ms
float delay_in = 2.;                // synaptic delay in ms
float delay_etoi = 2.;              // synaptic delay in ms
float delay_etoe = 5.;              // synaptic delay in ms
int hist_in, hist_etoi, hist_etoe;  // synaptic delay in multiples of dt
int hist_size;                      // size of delay buffer
float psi_in_noise = 0.002;         // amplitude of gaussian potential noise
float psi_ex_noise = 0.002;         // amplitude of gaussian potential noise
float psi_flow_noise = 0.005;       // amplitude of gaussian potential noise
float psi0_noise = 0.05;            // initial membrane potential noise
char psi0_iname[256] = "";          // name of potential input file

// Recurrent connections -------------------------------------------------------
float w_in = 40.;       // magnitude of recurrent inhibition
float w_etoi = 4.;      // magnitude of excitation to inhibitory neurons
float w_etoe = UNDEF;   // magnitude of excitation to excitatory neurons
float w_ratio;          // w_etoe / w_etoi
float l_in = 8.;        // inhibition lengthscale
float l_ex = 1.;        // excitation spread
float w_shift = 2.;     // shift for excitatory subpopulations

// Drive -----------------------------------------------------------------------
float a_in = 0.7;           // magnitude for inhibitory cells
float a_th = 0.2;           // theta magnitude for inhibitory cells
float a_in_rmag = 0.5;      // replay central magnitude for inhibitory cells
float a_in_rmin = UNDEF;    // replay shoulder minimum for inhibitory cells
float a_in_rcen = 0.;       // replay center radius for inhibitory cells
float a_in_rshoul = 0.7;    // replay shoulder radius for inhibitory cells

float a_ex_mag = 1.5;       // central magnitude for excitatory cells
float a_ex_min = UNDEF;     // shoulder minimum for excitatory cells
float a_ex_cen = 0.0;       // center radius for excitatory cells
float a_ex_shoul = 1.0;     // shoulder radius for excitatory cells
float a_ex_rmag = UNDEF;    // replay central magnitude for excitatory cells
float a_ex_rmin = UNDEF;    // replay shoulder minimum for excitatory cells
float a_ex_rcen = UNDEF;    // replay center radius for excitatory cells
float a_ex_rshoul = UNDEF;  // replay shoulder radius for excitatory cells

float vgain = 0.5;          // gain of velocity coupling

// Trajectory ------------------------------------------------------------------
char traj_iname[256] = "";    // name of trajectory input file
float vflow = 0.05;           // flow velocity in cm/ms or in 10 m/s
float angle_flow = UNDEF;     // flow angle

int xinit = 0;                // intial x position
int yinit = 0;                // intial y position
float vinit = 0.;             // initial speed in cm/ms or in 10 m/s
float angle_init = 0.;        // initial velocity angle 
float state_init = PI/2.;     // initial rat state (theta phase)

// Landmark --------------------------------------------------------------------
int tlearn = 500;               // time for learning landmark inputs
int tlm_flow = 300;             // flow time before learning landmark inputs
float lm_cutoff = 0.5;          // spiking cutoff factor
float a_ex_lmmin = UNDEF;       // landmark input minimum
float a_ex_lmmag = UNDEF;       // landmark input magnitude
char traj_lm_iname[256] = "";   // name of trajectory input file

// Arrays ----------------------------------------------------------------------
float **psi;     // membrane potentials
char **s;        // spikes

float **a;       // current drive to all neurons
float *a_ex;     // baseline drive to excitatory neurons
float *a_ex_r;   // drive to excitatory neurons during replay before learning
float *a_in_r;   // drive to inhibitory neurons during replay

float **a_ex_lm;            // drive to excitatory neurons during replay
int **lm;                   // landmark spike counts

fftwf_complex **w_fourier;  // recurrent kernels fourier space

float *s_temp;              // spikes for transformation real space
fftwf_complex *sw_fourier;  // spikes convolved with kernels fourier space
float *sw_temp;             // convolved spikes transformed back to real space
float ***sw;                // buffer containing recent convolved spike history
float *sw_to_in, *sw_to_ex; // for adding to different populations

// fftw plans ------------------------------------------------------------------
fftwf_plan s_forward;     // Neural sheet space to fourier space
fftwf_plan sw_reverse;    // Fourier space to neural sheet space

// Output parameters -----------------------------------------------------------
int out_psi = 1;     // Output potential file every tlap iterations
int out_s = 1;       // Output spike file every tlap iterations
int sbits = 128;     // Have binary spike outputs in multiples of sbits bits


static inline int fmini (int x, int y)
{
  return (x<y)?x:y;
}

static inline int fmaxi (int x, int y)
{
  return (x>y)?x:y;
}

static inline int nmod (int x, int y)
{
   int ret = x % y;
   if (ret < 0)
     ret += y;
   return ret;
}


//==============================================================================
// END Parameters and default value declarations
//==============================================================================


//==============================================================================
// Parameter input and output
//==============================================================================

// Parsing command line arguments
void get_parameters (int argc, char *argv[]) {

  int narg = 1;

  while (narg < argc) {
    if (!strcmp(argv[narg],"-fileroot")) {
      sscanf(argv[narg+1],"%s",fileroot);
      narg += 2;
    } else if (!strcmp(argv[narg],"-randseed")) {
      sscanf(argv[narg+1],"%d",&randseed);
      narg += 2;
    } 
    
    else if (!strcmp(argv[narg],"-laps_sim")) {
      sscanf(argv[narg+1],"%d",&laps_sim);
      narg += 2;
    } else if (!strcmp(argv[narg],"-laps_flow")) {
      sscanf(argv[narg+1],"%d",&laps_flow);
      narg += 2;
    } else if (!strcmp(argv[narg],"-tflow1")) {
      sscanf(argv[narg+1],"%d",&tflow1);
      narg += 2;
    } else if (!strcmp(argv[narg],"-tflow2")) {
      sscanf(argv[narg+1],"%d",&tflow2);
      narg += 2;
    } else if (!strcmp(argv[narg],"-trun")) {
      sscanf(argv[narg+1],"%d",&trun);
      narg += 2;
    } else if (!strcmp(argv[narg],"-treplay")) {
      sscanf(argv[narg+1],"%d",&treplay);
      narg += 2;
    } else if (!strcmp(argv[narg],"-tscreen")) {
      sscanf(argv[narg+1],"%d",&tscreen);
      narg += 2;
    } 
    
    else if (!strcmp(argv[narg],"-periodic")) {
      periodic = 1;
      narg++;
    } else if (!strcmp(argv[narg],"-n")) {
      sscanf(argv[narg+1],"%d",&n);
      narg += 2;
    } else if (!strcmp(argv[narg],"-np")) {
      sscanf(argv[narg+1],"%d",&np);
      narg += 2;
    } else if (!strcmp(argv[narg],"-dt")) {
      sscanf(argv[narg+1],"%f",&dt);
      narg += 2;
    } else if (!strcmp(argv[narg],"-tau_in")) {
      sscanf(argv[narg+1],"%f",&tau_in);
      narg += 2;
    } else if (!strcmp(argv[narg],"-tau_ex")) {
      sscanf(argv[narg+1],"%f",&tau_ex);
      narg += 2;
    } else if (!strcmp(argv[narg],"-delay_in")) {
      sscanf(argv[narg+1],"%f",&delay_in);
      narg += 2;
    } else if (!strcmp(argv[narg],"-delay_etoi")) {
      sscanf(argv[narg+1],"%f",&delay_etoi);
      narg += 2;
    } else if (!strcmp(argv[narg],"-delay_etoe")) {
      sscanf(argv[narg+1],"%f",&delay_etoe);
      narg += 2;
    } else if (!strcmp(argv[narg],"-psi_noise")) {
      sscanf(argv[narg+1],"%f",&psi_in_noise);
      psi_ex_noise = psi_in_noise;
      narg += 2;
    } else if (!strcmp(argv[narg],"-psi_in_noise")) {
      sscanf(argv[narg+1],"%f",&psi_in_noise);
      narg += 2;
    } else if (!strcmp(argv[narg],"-psi_ex_noise")) {
      sscanf(argv[narg+1],"%f",&psi_ex_noise);
      narg += 2;
    } else if (!strcmp(argv[narg],"-psi_flow_noise")) {
      sscanf(argv[narg+1],"%f",&psi_flow_noise);
      narg += 2;
    } else if (!strcmp(argv[narg],"-psi0_noise")) {
      sscanf(argv[narg+1],"%f",&psi0_noise);
      narg += 2;
    } else if (!strcmp(argv[narg],"-psi0")) {
      sscanf(argv[narg+1],"%s",psi0_iname);
      narg += 2;
    } 
    
    else if (!strcmp(argv[narg],"-w_in")) {
      sscanf(argv[narg+1],"%f",&w_in);
      narg += 2;
    } else if (!strcmp(argv[narg],"-w_etoi")) {
      sscanf(argv[narg+1],"%f",&w_etoi);
      narg += 2;
    } else if (!strcmp(argv[narg],"-w_etoe")) {
      sscanf(argv[narg+1],"%f",&w_etoe);
      narg += 2;
    } else if (!strcmp(argv[narg],"-l_in")) {
      sscanf(argv[narg+1],"%f",&l_in);
      narg += 2;
    } else if (!strcmp(argv[narg],"-l_ex")) {
      sscanf(argv[narg+1],"%f",&l_ex);
      narg += 2;
    } else if (!strcmp(argv[narg],"-w_shift")) {
      sscanf(argv[narg+1],"%f",&w_shift);
      narg += 2;
    } 
    
    else if (!strcmp(argv[narg],"-a_in")) {
      sscanf(argv[narg+1],"%f",&a_in);
      narg += 2;
    } else if (!strcmp(argv[narg],"-a_th")) {
      sscanf(argv[narg+1],"%f",&a_th);
      narg += 2;
    } else if (!strcmp(argv[narg],"-a_in_rmag")) {
      sscanf(argv[narg+1],"%f",&a_in_rmag);
      narg += 2;
    } else if (!strcmp(argv[narg],"-a_in_rmin")) {
      sscanf(argv[narg+1],"%f",&a_in_rmin);
      narg += 2;
    } else if (!strcmp(argv[narg],"-a_in_rcen")) {
      sscanf(argv[narg+1],"%f",&a_in_rcen);
      narg += 2;
    } else if (!strcmp(argv[narg],"-a_in_rshoul")) {
      sscanf(argv[narg+1],"%f",&a_in_rshoul);
      narg += 2;
    }
    
    else if (!strcmp(argv[narg],"-a_ex_mag")) {
      sscanf(argv[narg+1],"%f",&a_ex_mag);
      narg += 2;
    } else if (!strcmp(argv[narg],"-a_ex_min")) {
      sscanf(argv[narg+1],"%f",&a_ex_min);
      narg += 2;
    } else if (!strcmp(argv[narg],"-a_ex_cen")) {
      sscanf(argv[narg+1],"%f",&a_ex_cen);
      narg += 2;
    } else if (!strcmp(argv[narg],"-a_ex_shoul")) {
      sscanf(argv[narg+1],"%f",&a_ex_shoul);
      narg += 2;
    } else if (!strcmp(argv[narg],"-a_ex_rmag")) {
      sscanf(argv[narg+1],"%f",&a_ex_rmag);
      narg += 2;
    } else if (!strcmp(argv[narg],"-a_ex_rmin")) {
      sscanf(argv[narg+1],"%f",&a_ex_rmin);
      narg += 2;
    } else if (!strcmp(argv[narg],"-a_ex_rcen")) {
      sscanf(argv[narg+1],"%f",&a_ex_rcen);
      narg += 2;
    } else if (!strcmp(argv[narg],"-a_ex_rshoul")) {
      sscanf(argv[narg+1],"%f",&a_ex_rshoul);
      narg += 2;
    } else if (!strcmp(argv[narg],"-vgain")) {
      sscanf(argv[narg+1],"%f",&vgain);
      narg += 2;
    } 
    
    else if (!strcmp(argv[narg],"-traj")) {
      sscanf(argv[narg+1],"%s",traj_iname);
      narg += 2;
    } else if (!strcmp(argv[narg],"-vflow")) {
      sscanf(argv[narg+1],"%f",&vflow);
      narg += 2;
    } else if (!strcmp(argv[narg],"-angle_flow")) {
      sscanf(argv[narg+1],"%f",&angle_flow);
      narg += 2;
    } else if (!strcmp(argv[narg],"-xinit")) {
      sscanf(argv[narg+1],"%d",&xinit);
      narg += 2;
    } else if (!strcmp(argv[narg],"-yinit")) {
      sscanf(argv[narg+1],"%d",&yinit);
      narg += 2;
    } else if (!strcmp(argv[narg],"-vinit")) {
      sscanf(argv[narg+1],"%f",&vinit);
      vflow = vinit;
      narg += 2;
    } else if (!strcmp(argv[narg],"-angle_init")) {
      sscanf(argv[narg+1],"%f",&angle_init);
      narg += 2;
    } else if (!strcmp(argv[narg],"-state_init")) {
      sscanf(argv[narg+1],"%f",&state_init);
      narg += 2;
    }

    else if (!strcmp(argv[narg],"-tlearn")) {
      sscanf(argv[narg+1],"%d",&tlearn);
      narg += 2;
    } else if (!strcmp(argv[narg],"-tlm_flow")) {
      sscanf(argv[narg+1],"%d",&tlm_flow);
      narg += 2;
    } else if (!strcmp(argv[narg],"-a_ex_lmmin")) {
      sscanf(argv[narg+1],"%f",&a_ex_lmmin);
      narg += 2;
    } else if (!strcmp(argv[narg],"-a_ex_lmmag")) {
      sscanf(argv[narg+1],"%f",&a_ex_lmmag);
      narg += 2;
    } else if (!strcmp(argv[narg],"-traj_lm")) {
      sscanf(argv[narg+1],"%s",traj_lm_iname);
      narg += 2;
    } else if (!strcmp(argv[narg],"-lm_cutoff")) {
      sscanf(argv[narg+1],"%f",&lm_cutoff);
      narg += 2;
    }
    
    else if (!strcmp(argv[narg],"-no_psi")) {
      out_psi = 0;
      narg++;
    } else if (!strcmp(argv[narg],"-no_s")) {
      out_s = 0;
      narg++;
    } else if (!strcmp(argv[narg],"-sbits")) {
      sscanf(argv[narg+1],"%d",&sbits);
      narg += 2;
    }
    
    else {
      printf("unknown option: %s\n", argv[narg]);
      exit(0);
    }
  }  

}


// Printing parameter values to stdout
void print_parameters () {

  char name[256];
  FILE *fo;

  sprintf(name, "%s_params.txt", fileroot);
  fo = fopen(name, "w");


  fprintf(fo, "randseed        = %d\n", randseed);
  fprintf(fo, "\n");

  fprintf(fo, "laps_sim        = %d\n", laps_sim);
  fprintf(fo, "laps_flow       = %d\n", laps_flow);
  fprintf(fo, "tflow1          = %d\n", tflow1);
  fprintf(fo, "tflow2          = %d\n", tflow2);
  fprintf(fo, "trun            = %d\n", trun);
  fprintf(fo, "treplay         = %d\n", treplay);
  fprintf(fo, "\n");

  fprintf(fo, "periodic        = %d\n", periodic);
  fprintf(fo, "n               = %d\n", n);
  fprintf(fo, "np              = %d\n", np);
  fprintf(fo, "dt              = %f\n", dt);
  fprintf(fo, "tau_in          = %f\n", tau_in);
  fprintf(fo, "tau_ex          = %f\n", tau_ex);
  fprintf(fo, "delay_in        = %f\n", delay_in);
  fprintf(fo, "delay_etoi      = %f\n", delay_etoi);
  fprintf(fo, "delay_etoe      = %f\n", delay_etoe);
  fprintf(fo, "psi_in_noise    = %f\n", psi_in_noise);
  fprintf(fo, "psi_ex_noise    = %f\n", psi_ex_noise);
  fprintf(fo, "psi_flow_noise  = %f\n", psi_flow_noise);
  fprintf(fo, "psi0_noise      = %f\n", psi0_noise);
  fprintf(fo, "psi0_iname      = %s\n", psi0_iname);
  fprintf(fo, "\n");

  fprintf(fo, "w_in            = %f\n", w_in);
  fprintf(fo, "w_etoi          = %f\n", w_etoi);
  fprintf(fo, "w_etoe          = %f\n", w_etoe);
  fprintf(fo, "l_in            = %f\n", l_in);
  fprintf(fo, "l_ex            = %f\n", l_ex);
  fprintf(fo, "w_shift         = %f\n", w_shift);
  fprintf(fo, "\n");

  fprintf(fo, "a_in            = %f\n", a_in);
  fprintf(fo, "a_th            = %f\n", a_th);
  fprintf(fo, "a_in_rmag       = %f\n", a_in_rmag);
  fprintf(fo, "a_in_rmin       = %f\n", a_in_rmin);
  fprintf(fo, "a_in_rcen       = %f\n", a_in_rcen);
  fprintf(fo, "a_in_rshoul     = %f\n", a_in_rshoul);
  fprintf(fo, "\n");

  fprintf(fo, "a_ex_mag        = %f\n", a_ex_mag);
  fprintf(fo, "a_ex_min        = %f\n", a_ex_min);
  fprintf(fo, "a_ex_cen        = %f\n", a_ex_cen);
  fprintf(fo, "a_ex_shoul      = %f\n", a_ex_shoul);
  fprintf(fo, "a_ex_rmag       = %f\n", a_ex_rmag);
  fprintf(fo, "a_ex_rmin       = %f\n", a_ex_rmin);
  fprintf(fo, "a_ex_rcen       = %f\n", a_ex_rcen);
  fprintf(fo, "a_ex_rshoul     = %f\n", a_ex_rshoul);
  fprintf(fo, "vgain           = %f\n", vgain);
  fprintf(fo, "\n");

  fprintf(fo, "traj_iname      = %s\n", traj_iname);
  fprintf(fo, "vflow           = %f\n", vflow);
  fprintf(fo, "angle_flow      = %f\n", angle_flow);
  fprintf(fo, "xinit           = %d\n", xinit);
  fprintf(fo, "yinit           = %d\n", yinit);
  fprintf(fo, "vinit           = %f\n", vinit);
  fprintf(fo, "angle_init      = %f\n", angle_init);
  fprintf(fo, "state_init      = %f\n", state_init);
  fprintf(fo, "\n");

  fprintf(fo, "tlearn          = %d\n", tlearn);
  fprintf(fo, "tlm_flow        = %d\n", tlm_flow);
  fprintf(fo, "a_ex_lmmag      = %f\n", a_ex_lmmag);
  fprintf(fo, "a_ex_lmmin      = %f\n", a_ex_lmmin);
  fprintf(fo, "traj_lm_iname   = %s\n", traj_lm_iname);
  fprintf(fo, "lm_cutoff       = %f\n", lm_cutoff);
  fprintf(fo, "\n");

  fprintf(fo, "sbits           = %d\n", sbits);
  fprintf(fo, "n2bin           = %d\n", n2bin);

  fclose(fo);

}


void output_float_list(float *list, int idim, char *filename) {

  char name[256];
  FILE *fo;
  int i;

  sprintf(name, "%s_%s.txt", fileroot, filename);
  fo = fopen(name, "w");

  for (i = 0; i < idim; i++)
    fprintf(fo, "%f ", list[i]); 

  fclose(fo);
}

void output_complex_list(fftwf_complex *list, int idim, char *filename) {

  char name[256];
  FILE *fo;
  int i;

  sprintf(name, "%s_%s.txt", fileroot, filename);
  fo = fopen(name, "w");

  for (i = 0; i < idim; i++)
    fprintf(fo, "%f%+fI ", crealf(list[i]), cimagf(list[i])); 

  fclose(fo);
}

void output_float_array(float **arr, int idim, int jdim, char *filename) {
  
  char name[256];
  FILE *fo;
  int i, j;

  sprintf(name, "%s_%s.txt", fileroot, filename);
  fo = fopen(name, "w");

  for (i = 0; i < idim; i++) {
    for (j = 0; j < jdim; j++) 
      fprintf(fo, "%f ", arr[i][j]); 
    fprintf(fo, "\n");
  }
  fclose(fo);
}


//==============================================================================
// END Parameter input and output
//==============================================================================


//==============================================================================
// Network structure setup
//==============================================================================


// Setup durations for laps
void setup_simulation_durations () {

  // Reading in durations from trajectory filename
  if (trun == UNDEF || treplay == UNDEF) {

    strtok(basename(traj_iname), "_");
    
    trun    = atoi(strtok(NULL, "-"));
    treplay = atoi(strtok(NULL, "-"));

    printf("Durations %d run and %d replay from traj filename\n",
                                         trun, treplay); fflush(stdout);
  
  }

  tlap = trun + treplay;

}


// Setup neural sheet size
void setup_network_dimensions () {

  int pad_min;

  // FFTW works fastest when np has small prime factors, so setting np is
  // required
  if (periodic) {

    n = np;
    pad = 0;

  } else {

    // Minimum amount of zero-padding required by recurrent and coupling kernels
    pad_min = fmaxf(floor(2*l_ex) + w_shift, floor(2*l_in));
    if (np <= pad_min) {
      printf("Must set padded size np larger than %d\n", pad_min);
        fflush(stdout);
      exit(0);
    }

    // If n not set, calculating largest valid n given np size
    if (n < 0) {
      n = np - pad_min;
      if (n % 2 == 1)
        n--;
      pad = np - n;
      printf("Defining n in terms of np\n"); fflush(stdout);
    } else if (n > np - pad_min) {
      printf("Unpadded size n too large for padded size np\n"); fflush(stdout);
      exit(0);
    }
  
  }

  n2 = n * n;
  np2 = np * np;
  npfft = np * (np/2 + 1);

  if (sbits % 8 != 0) {
    printf("sbits must be a multiple of 8\n"); fflush(stdout);
    exit(0);
  }
  n2bin = (sbits * (n2/sbits + (n2%sbits != 0))) / 8;   // ceiling of n2 as a
                                                        // multiple of sbits
                                                        // divided by 8

}


// Setup neuron arrays and activities
void setup_neurons () {

  FILE *psi0_i;
  float *dummy;
  int c, p;

  psi = malloc(5 * sizeof(float *));
  for (p = 0; p < 5; p++)
    psi[p] = malloc(n2 * sizeof(float));
  
  // Initialize potentials randomly or read them in
  if (psi0_iname[0] == '\0')
    for (p = 0; p < 5; p++)
      for (c = 0; c < n2; c++)
        psi[p][c] = drand48()
                    + psi0_noise * r4_nor(&randseed,r4kn,r4fn,r4wn);
  else {

    for (c = 0; c < n2; c++)
      psi[0][c] = a_in;

    if (NULL == (psi0_i = fopen(psi0_iname, "r"))) {
      printf("error opening potential file %s\n", psi0_iname); fflush(stdout);
      exit(0);
    }
    printf("potential file opened successfully\n");

    for (c = 0; c < n2; c++)
      if (fscanf(psi0_i, "%f ", &psi[1][c]) != 1) {
        printf("error: end of potential file\n"); fflush(stdout);
        exit(0);
      }

    if (fscanf(psi0_i, "%f", dummy) != EOF) {
      printf("error: potential file size does not match simulation size");
        fflush(stdout);
      exit(0);
    }

    fclose(psi0_i);

    for (p = 2; p < 5; p++)
      for (c = 0; c < n2; c++) 
        psi[p][c] = psi[1][c];

    for (p = 0; p < 5; p++)
      for (c = 0; c < n2; c++)
        psi[p][c] += psi0_noise * r4_nor(&randseed,r4kn,r4fn,r4wn);
  }

  s = malloc(5 * sizeof(char **));
  for (p = 0; p < 5; p++) {
    s[p] = malloc(n2 * sizeof(char));
    for (c = 0; c < n2; c++) {
      s[p][c] = 0;
    }
  }

}


// Calculating input drive
void calculate_drive (
    float *a,
    float a_mag, float a_min, float a_cen, float a_shoul
) {

  int i, j, c;
  float i_scaled[n], j_scaled[n];
  float r;

  // Setup scaled coordinates
  for (i = 0; i < n; i++)
    i_scaled[i] = (i-n/2.+0.5)/(n/2.);
  for (j = 0; j < n; j++)
    j_scaled[j] = (j-n/2.+0.5)/(n/2.);

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      c = n * i + j;

      r = sqrt(i_scaled[i]*i_scaled[i] + j_scaled[j]*j_scaled[j]);

      if (r >= a_shoul)
        a[c] = a_min;
      else if (r <= a_cen)
        a[c] = a_mag;
      else
        a[c] = a_min + (a_mag - a_min)
                    * ((1 + cos(PI * (r-a_cen)/(a_shoul-a_cen))) / 2);
    
    }

}


// Setup input drives
void setup_drive () {

  int p;

  a = malloc(5 * sizeof(float *));
  for (p = 0; p < 5; p++)
    a[p] = malloc(n2 * sizeof(float));


  a_in_r = malloc(n2 * sizeof(float));
  if (a_in_rmin == UNDEF)
    a_in_rmin = a_in_rmag;
  calculate_drive(a_in_r, a_in_rmag, a_in_rmin, a_in_rcen, a_in_rshoul);

  output_float_list(a_in_r, n2, "a-in-r");

  
  a_ex = malloc(n2 * sizeof(float));
  if (a_ex_min == UNDEF)
    a_ex_min = a_ex_mag;
  calculate_drive(a_ex, a_ex_mag, a_ex_min, a_ex_cen, a_ex_shoul);

  output_float_list(a_ex, n2, "a-ex");


  a_ex_r = malloc(n2 * sizeof(float));
  if (a_ex_rmag == UNDEF)
    a_ex_rmag = a_ex_mag;
  if (a_ex_rmin == UNDEF)
    a_ex_rmin = a_ex_rmag;
  if (a_ex_rcen == UNDEF)
    a_ex_rcen = a_ex_cen;
  if (a_ex_rshoul == UNDEF)
    a_ex_rshoul = a_ex_shoul;
  calculate_drive(a_ex_r, a_ex_rmag, a_ex_rmin, a_ex_rcen, a_ex_rshoul);

  output_float_list(a_ex_r, n2, "a-ex-r");

}


// Calculate recurrent inhibition kernel
void calculate_recurrent_in (float *w, float w_mag, float l) { 

  int i, j, b;
  float shifted[np];
  float r;


  // Setup shifted coordinates
  for (i = 0; i < np/2; i++)
    shifted[i] = i;
  for (; i < np; i++)
    shifted[i] = i - np;


  for (i = 0; i < np; i++)
    for (j = 0; j < np; j++) {
      b = np * i + j;

      r = sqrt(shifted[i]*shifted[i] + shifted[j]*shifted[j]);

      if (r < 2 * l)
        w[b] = -(w_mag / (l*l)) * ((1 - cos(PI*r/l)) / 2);
      else
        w[b] = 0.;
  
  }

}


// Calculate recurrent excitation kernel
void calculate_recurrent_ex (
    float *w, float w_mag, float l,
    float i_shift, float j_shift
) { 

  int i, j, b;
  float i_shifted[np], j_shifted[np];
  float r;


  // Setup shifted coordinates
  for (i = 0; i < np/2; i++)
    i_shifted[i] = i - i_shift;
  for (; i < np; i++)
    i_shifted[i] = i - i_shift - np;
  for (j = 0; j < np/2; j++)
    j_shifted[j] = j - j_shift;
  for (; j < np; j++)
    j_shifted[j] = j - j_shift - np;


  for (i = 0; i < np; i++)
    for (j = 0; j < np; j++) {
      b = np * i + j;

      r = sqrt(i_shifted[i]*i_shifted[i] + j_shifted[j]*j_shifted[j]);

      if (r < 2 * l)
        w[b] = (w_mag / (4*l*l)) * ((1 + cos(PI*r/(2*l))) / 2);
      else
        w[b] = 0.;
  
  }

}


// Setup recurrent connections
void setup_recurrent () {

  int b, p;
  
  float **w;
  fftwf_plan *w_forward;


  if (w_etoi == 0.) {
    printf("cannot have w_etoi = 0.\n"); fflush(stdout);
    exit(0);
  }
  if (w_etoe == UNDEF)
    w_etoe = w_etoi;
  w_ratio = w_etoe / w_etoi;

  // Allocate space for the recurrent kernels and fourier transforms
  w = malloc(5 * sizeof(float *));
  for (p = 0; p < 5; p++)
    w[p] = malloc(np2 * sizeof(float));

  w_fourier = malloc(5 * sizeof(fftwf_complex *));
  for (p = 0; p < 5; p++)
    w_fourier[p] = malloc(npfft * sizeof(fftwf_complex));
  
  // Create transform plans (may overwrite w)
  w_forward = malloc(5 * sizeof(fftwf_plan));
  for (p = 0; p < 5; p++)
    w_forward[p] = fftwf_plan_dft_r2c_2d(np, np, w[p], w_fourier[p], FFTW_ESTIMATE);


  // Calculate the recurrent kernels (after planning)
  calculate_recurrent_in(w[0], w_in, l_in);
  calculate_recurrent_ex(w[1], w_etoi, l_ex, -w_shift,       0.);
  calculate_recurrent_ex(w[2], w_etoi, l_ex,  w_shift,       0.);
  calculate_recurrent_ex(w[3], w_etoi, l_ex,       0., -w_shift);
  calculate_recurrent_ex(w[4], w_etoi, l_ex,       0.,  w_shift);

  output_float_list(w[0], np2, "w-i");
  output_float_list(w[1], np2, "w-l");


  // Perform the fourier transforms
  for (p = 0; p < 5; p++) {

    fftwf_execute(w_forward[p]);
    fftwf_destroy_plan(w_forward[p]);

    // Normalizing
    for (b = 0; b < npfft; b++)
      w_fourier[p][b] /= np2;

  }

  // After obtaining its fourier transforms, we don't need w anymore
  for (p = 0; p < 5; p++)
    free(w[p]);
  free(w);

  free(w_forward);

}


// Setup fourier transform plans for spike convolutions
void setup_convolution () {

  int b, p;
  int t;


  printf("Creating fourier transform plans... "); fflush(stdout);

  // Calculate length for history buffers
  hist_in   = fmaxi((int)round(delay_in   / dt), 1);
  hist_etoi = fmaxi((int)round(delay_etoi / dt), 1);
  hist_etoe = fmaxi((int)round(delay_etoe / dt), 1);
  hist_size = fmaxi(fmaxi(hist_in, hist_etoi), hist_etoe);

  // Allocate space for fourier transform kernels
  s_temp = malloc(np2 * sizeof(float));
  sw_fourier = malloc(npfft * sizeof(fftwf_complex));
  sw_temp = malloc(np2 * sizeof(float));

  // Allocate space for convolved spike history buffers
  sw = malloc(hist_size * sizeof(float **));
  for (t = 0; t < hist_size; t++) {
    sw[t] = malloc(5 * sizeof(float *));
    for (p = 0; p < 5; p++)
      sw[t][p] = malloc(np2 * sizeof(float));
  }

  // Allocate space for spikes convolved with kernels
  sw_to_ex = malloc(n2 * sizeof(float));
  sw_to_in = malloc(n2 * sizeof(float));


  // Create forward plans
  s_forward = fftwf_plan_dft_r2c_2d(np, np, s_temp, sw_fourier, FFTW_PATIENT);
  
  // Create reverse plan
  sw_reverse = fftwf_plan_dft_c2r_2d(np, np, sw_fourier, sw_temp, FFTW_PATIENT);


  // Initializing temporary spike buffer and convolved spike history buffer to 0.
  for (b = 0; b < np2; b++)
    s_temp[b] = 0.;
  for (t = 0; t < hist_size; t++)
    for (p = 0; p < 5; p++)
      for (b = 0; b < np2; b++)
        sw[t][p][b] = 0.;


  printf("done\n"); fflush(stdout);
  
}


//==============================================================================
// END Network structure setup
//==============================================================================


//==============================================================================
// Network dynamics
//==============================================================================


// Update drive to inhibitory population
void update_drive_in (float *a, float state) {

  int c;
  float taper_fac, th_fac;

  if (state >= 0.) {  // run
    th_fac = -cos(state);
    for (c = 0; c < n2; c++)
      a[c] = a_in + a_th * th_fac;
  } else if (state >= -1.) {  // replay
    taper_fac = fminf(-state, 1.);
    for (c = 0; c < n2; c++)
      a[c] = taper_fac * a_in_r[c] + (1.-taper_fac) * a_in;
  } else  // landmark
    for (c = 0; c < n2; c++)
      a[c] = a_in;

}


// Update drive to one excitatory population
void update_drive_ex (float *a, float vgain_fac, float state, int end) {

  int c;
  float taper_fac;

  if (state >= 0.)  // run
    for (c = 0; c < n2; c++)
      a[c] = a_ex[c] * vgain_fac;
  else if (state >= -1.) {  // replay
    taper_fac = fminf(-state, 1.);
    for (c = 0; c < n2; c++)
      a[c] = taper_fac * a_ex_r[c] + (1.-taper_fac) * a_ex[c];
  } else {  // landmark
    if (end < 0 || end > 2) {
      printf("error in update_drive_ex: end must be 0, 1, or 2\n");
        fflush(stdout);
      exit(0);
    }
    for (c = 0; c < n2; c++)
      a[c] = a_ex_lm[end][c];
  }

}


// Update neuron activity
void update_activity (int *hist_now, int flow_noise) {

  int i, j, b, c, p;
  int past_in, past_etoi, past_etoe;


  past_in   = nmod(*hist_now - hist_in  , hist_size);
  past_etoi = nmod(*hist_now - hist_etoi, hist_size);
  past_etoe = nmod(*hist_now - hist_etoe, hist_size);

  // Retrieve spike convolutions from the appropriate history register
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      c = n * i + j;
      b = np * i + j;

      sw_to_in[c] =   sw[past_etoi][1][b] + sw[past_etoi][2][b]
                    + sw[past_etoi][3][b] + sw[past_etoi][4][b];

      sw_to_ex[c] = sw[past_in][0][b]
                    + w_ratio * (  sw[past_etoe][1][b] + sw[past_etoe][2][b]
                                 + sw[past_etoe][3][b] + sw[past_etoe][4][b]);
    }

  // Excitatory drive and recurrent inhibition
  for (c = 0; c < n2; c++)
      psi[0][c] += (sw_to_in[c] + dt * (a[0][c] - psi[0][c])) / tau_in;
  for (p = 1; p < 5; p++)
    for (c = 0; c < n2; c++)
      psi[p][c] += (sw_to_ex[c] + dt * (a[p][c] - psi[p][c])) / tau_ex;

  // Add noise
  if (flow_noise)
    for (p = 0; p < 5; p++)
      for (c = 0; c < n2; c++)
        psi[p][c] += psi_flow_noise * r4_nor(&randseed,r4kn,r4fn,r4wn);
  else {
    if (psi_in_noise > 0.)
        for (c = 0; c < n2; c++)
          psi[0][c] += psi_in_noise * r4_nor(&randseed,r4kn,r4fn,r4wn);
    if (psi_ex_noise > 0.)
      for (p = 1; p < 5; p++)
        for (c = 0; c < n2; c++)
          psi[p][c] += psi_ex_noise * r4_nor(&randseed,r4kn,r4fn,r4wn);
  }

  for (p = 0; p < 5; p++) {

    // Reset potentials and update spikes
    for (i = 0; i < n; i++)
      for (j = 0; j < n; j++) {

        c = n * i + j;
        b = np * i + j;

        if (psi[p][c] >= 1.) {
          psi[p][c] = 0.;
          s[p][c] = 1;
          s_temp[b] = 1.;
        } else {
          if (psi[p][c] < -1.)
            psi[p][c] = -1.;
          s[p][c] = 0;
          s_temp[b] = 0.;
        }

      }
  

    // Convolve spikes with recurrent kernels and load into the proper sw
    // register
    fftwf_execute(s_forward);
    for (b = 0; b < npfft; b++)
      sw_fourier[b] *= w_fourier[p][b];
    fftwf_execute(sw_reverse);
    memcpy(sw[*hist_now][p], sw_temp, np2 * sizeof(float));

  }

  *hist_now = (*hist_now + 1) % hist_size;

}


//==============================================================================
// END Network dynamics
//==============================================================================


//==============================================================================
// Animal trajectory
//==============================================================================


// Setup trajectory input file
void setup_trajectory (FILE **traj_i) {

  if (NULL == (*traj_i = fopen(traj_iname, "r"))) {
    printf("error opening trajectory file %s\n", traj_iname); fflush(stdout);
    exit(0);
  }
  printf("trajectory file opened successfully\n");

}


// Read in velocity and update position and drives
void update_trajectory (
    FILE *traj_i,
    float *vx,     float *vy,     float *state,
    float *vx_old, float *vy_old, float *state_old,
    float *xpos, float *ypos, int end
) {

  bool state_to_v;

  if (fscanf(traj_i, "%f %f %f", vx, vy, state) != 3) {
    printf("end of trajectory file. returning to beginning...\n"); fflush(stdout);
    rewind(traj_i);
    if (fscanf(traj_i, "%f %f %f", vx, vy, state) != 3) {
      printf("error reading trajectory file\n"); fflush(stdout);
      exit(0);
    }
  }
  
  *xpos += (*vx) * dt;
  *ypos += (*vy) * dt;

  state_to_v = *state != *state_old && (*state <= 0. || *state_old <= 0.);

  if (*state != *state_old)
    update_drive_in(a[0], *state);
  if (*vx != *vx_old || state_to_v) {
    update_drive_ex(a[1], 1. - vgain * (*vx), *state, end);
    update_drive_ex(a[2], 1. + vgain * (*vx), *state, end);
  }
  if (*vy != *vy_old || state_to_v) {
    update_drive_ex(a[3], 1. - vgain * (*vy), *state, end);
    update_drive_ex(a[4], 1. + vgain * (*vy), *state, end);
  }

  *vx_old = *vx;
  *vy_old = *vy;
  *state_old = *state;

}


//==============================================================================
// END Animal trajectory
//==============================================================================


//==============================================================================
// Landmark learning
//==============================================================================


// Setup landmark input arrays
void setup_landmark (FILE **traj_i) {

  int c, end;

  if (NULL == (*traj_i = fopen(traj_lm_iname, "r"))) {
    printf("error opening landmark learning traj file %s\n", traj_lm_iname);
      fflush(stdout);
    exit(0);
  }
  printf("landmark trajectory file opened successfully\n");

  lm = malloc(2 * sizeof(int *));
  for (end = 0; end < 2; end++) {
    lm[end] = malloc(n2 * sizeof(int));
    for (c = 0; c < n2; c++)
      lm[end][c] = 0;
  } 

  a_ex_lm = malloc(3 * sizeof(float *));
  for (end = 0; end < 3; end++) {
    a_ex_lm[end] = malloc(n2 * sizeof(float));
    for (c = 0; c < n2; c++)
      a_ex_lm[end][c] = 0.;
  } 

  if (a_ex_lmmin == UNDEF)
    a_ex_lmmin = periodic ? a_ex_rmin : a_ex_min;
  if (a_ex_lmmag == UNDEF)
    a_ex_lmmag = a_ex_mag;

}


// Update drive to all populations for landmark learning
void switch_drive_landmark () {

  int p;

  // Inhibitory neurons take run values
  update_drive_in(a[0], PI/2.);

  // Excitatory neurons take landmark values
  for (p = 1; p < 5; p++)
    update_drive_ex(a[p], 1., PI/2., 2);

}


// Add new spikes from excitatory neurons to landmark counter
void update_landmark_counts (int end) {

  int c, p;

  for (p = 1; p < 5; p++)
    for (c = 0; c < n2; c++)
      lm[end][c] += s[p][c];

}

// Calculate landmark inputs
void calculate_landmark_inputs () {

  int c, end;
  int lm_max;
  float lm_frac;

  for (end = 0; end < 2; end++) {

    lm_max = 0;
    for (c = 0; c < n2; c++)
      if (lm_max < lm[end][c])
        lm_max = lm[end][c];

    for (c = 0; c < n2; c++) {

      lm_frac = (float)lm[end][c] / lm_max;
      if (lm_frac > lm_cutoff)
        a_ex_lm[end][c] = a_ex_lmmin
          + (a_ex_lmmag-a_ex_lmmin) * (lm_frac-lm_cutoff) / (1.-lm_cutoff);
      else
        a_ex_lm[end][c] = a_ex_lmmin;

    }

  }

  if (periodic)
    calculate_drive(a_ex_lm[2], a_ex_lmmag, a_ex_lmmin, a_ex_cen, a_ex_shoul);
  else
    calculate_drive(a_ex_lm[2], a_ex_mag, a_ex_min, a_ex_cen, a_ex_shoul);

  output_float_array(a_ex_lm, 2, n2, "a-ex-lm");

}


//==============================================================================
// END Landmark learning
//==============================================================================


//==============================================================================
// Activity flow routines
//==============================================================================


// Flow the population activity at constant velocity
void flow_activity_const (
    float v, float angle,
    int tflow, int stage,
    int *hist_now
) {

  float vx, vy;
  int tnow;
 
  clock_t tic, toc;

  vx = v * cos(angle);
  vy = v * sin(angle);
  
  update_drive_in(a[0], PI/2.);
  update_drive_ex(a[1], 1. - vgain * vx, PI/2., -1);
  update_drive_ex(a[2], 1. + vgain * vx, PI/2., -1);
  update_drive_ex(a[3], 1. - vgain * vy, PI/2., -1);
  update_drive_ex(a[4], 1. + vgain * vy, PI/2., -1);

  printf("Constant flow stage %d:\n", stage); fflush(stdout);

  tic = clock();
  for (tnow = 1; tnow <= tflow; tnow++) {

    if (tscreen && tnow % tscreen == 0)
      printf("%d\n", tnow); fflush(stdout);

    update_activity(hist_now, 1);
    
  }
  toc = clock();

  printf("elapsed time for stage %d: %f seconds\n",
                                  stage, (double)(toc-tic)/CLOCKS_PER_SEC);
  
}


//==============================================================================
// END Activity flow routines
//==============================================================================


//==============================================================================
// Data output
//==============================================================================


// Output membrane potentials as plaintext
void output_potentials_text (int stage) {

  char name[256];
  FILE *psi_o;
  int c, p;

  sprintf(name, "%s_psi-setup%d.txt", fileroot, stage);
  
  psi_o = fopen(name, "w");
      
  for (p = 0; p < 5; p++) {
    for (c = 0; c < n2; c++) 
      fprintf(psi_o, "%1.2f ", psi[p][c]); 
    fprintf(psi_o, "\n");
  }

  fclose(psi_o);

}


// Output spikes as plaintext
void output_spikes_text (int stage) {

  char name[256];
  FILE *s_o;
  int c, p;

  sprintf(name, "%s_s-setup%d.txt", fileroot, stage);
  
  s_o = fopen(name, "w");
      
  for (p = 0; p < 5; p++) {
    for (c = 0; c < n2; c++) 
      fprintf(s_o, "%1d ", s[p][c]); 
    fprintf(s_o, "\n");
  }

  fclose(s_o);

}


// Output membrane potentials as binary
void output_potentials_binary (FILE **psi_o) {

  signed char *psi_buf;
  int c, p;

  psi_buf = malloc(n2 * sizeof(char));

  for (p = 0; p < 5; p++) {

    for (c = 0; c < n2; c++)
      psi_buf[c] = (signed char)fmaxf(50. * psi[p][c], -127.);

    fwrite(psi_buf, sizeof(char), n2, psi_o[p]);

  }

  free(psi_buf);

}


// Output spikes as binary
void output_spikes_binary (FILE **s_o) {

  unsigned char *s_buf;
  int c, p;

  s_buf = malloc(n2bin * sizeof(char));

  for (p = 0; p < 5; p++) {

    for (c = 0; c < n2bin; c++)
      s_buf[c] = 0;

    for (c = 0; c < n2; c++)
      s_buf[c/8] |= s[p][c] << (7 - (c % 8));

    fwrite(s_buf, sizeof(char), n2bin, s_o[p]);

  }
  
  free(s_buf);

}


// Open binary files
void open_binary (FILE **fo, char *type, int lap) {

  char name[256];
  int p;

  for (p = 0; p < 5; p++) {
    sprintf(name, "%s_%s%1d-lap%02d.dat", fileroot, type, p, lap);
    fo[p] = fopen(name, "wb");
  }
  
  printf("new binary output files for %s created\n", type);

}

// Close membrane potential output files
void close_binary (FILE **fo) {

  int p;

  for (p = 0; p < 5; p++)
    fclose(fo[p]);

}


//==============================================================================
// END Data output
//==============================================================================



int main(int argc, char *argv[]) {

  float xpos, ypos;
  float vx, vy, state, th_fac;     // state represents theta phase when positive
  float vx_old, vy_old, state_old; // and replay period taper when negative
  
  FILE *traj_i, *traj_lm_i, *traj_o;

  FILE **psi_o, **s_o;

  int hist_now;

  int lap, tnow;
  int ch, place;
  char name[256];


  get_parameters(argc, argv);
  
  if (strcmp(fileroot, "") == 0 ||
      fileroot[0] == '-' ||
      fileroot[strlen(fileroot)-1] == '/') {
    printf("Must set -fileroot correctly\n"); fflush(stdout);
    exit(0);
  }

  if (!randseed) {
    randseed = (uint32_t)(time(NULL) % 1048576);
    place = 1;
    for (ch = 0; ch < strlen(fileroot); ch++) {
      randseed += fileroot[ch] * place;
      place = (place * 128) % 100000;
    }
  }
  srand48(randseed);

  r4_nor_setup(r4kn,r4fn,r4wn);

  setup_simulation_durations();
  
  setup_network_dimensions();

  setup_neurons();

  setup_drive();

  setup_recurrent();

  setup_convolution();
  
  setup_trajectory(&traj_i);

  setup_landmark(&traj_lm_i);

  print_parameters();

  // Perform miscellaneous initializations
  vx_old = UNDEF;
  vy_old = UNDEF;
  state_old = UNDEF;

  xpos = xinit;
  ypos = yinit;

  hist_now = 0;

  // Setup output files
  sprintf(name, "%s_traj.txt", fileroot);
  traj_o = fopen(name, "w");
  if (out_psi)
    psi_o = malloc(5 * sizeof(FILE *));
  if (out_s)
    s_o = malloc(5 * sizeof(FILE *));
 



  printf("\nSTARTING SETUP\n"); fflush(stdout);

  output_potentials_text(1);
  output_spikes_text(1);

  // Flow neural population activity with constant velocities
  flow_activity_const(0., 0., tflow1, 1, &hist_now);

  output_potentials_text(2);
  output_spikes_text(2);
  
  if (angle_flow == UNDEF) {
    flow_activity_const(vflow, PI/2-PI/5, tflow2, 2, &hist_now);
    flow_activity_const(vflow,    2*PI/5, tflow2, 3, &hist_now);
    flow_activity_const(vflow,      PI/4, tflow2, 4, &hist_now);
  } else
    flow_activity_const(vflow, angle_flow, tflow2, 2, &hist_now);
  
  output_potentials_text(3);
  output_spikes_text(3);




  // Trajectory flow
  printf("Trajectory flow:\n"); fflush(stdout);

  for (lap = 1; lap <= laps_flow; lap++)
    for (tnow = 1; tnow <= trun; tnow++) {

      if (tscreen && tnow % tscreen == 0)
        printf("lap = %d, t = %d\n", lap, tnow); fflush(stdout);

      update_trajectory(traj_lm_i, &vx, &vy, &state,
                           &vx_old, &vy_old, &state_old, &xpos, &ypos, -1);

      update_activity(&hist_now, 0);
     
    }

  output_potentials_text(4);
  output_spikes_text(4);
 



  // Learning landmark inputs at both track ends
  printf("Learning landmark inputs:\n"); fflush(stdout);

  for (lap = 1; lap <= 2; lap++) {

    for (tnow = 1; tnow <= trun; tnow++) {

      if (tscreen && tnow % tscreen == 0)
        printf("lap = %d, run t = %d\n", lap, tnow); fflush(stdout);
      update_trajectory(traj_lm_i, &vx, &vy, &state,
                           &vx_old, &vy_old, &state_old, &xpos, &ypos, -1);
      update_activity(&hist_now, 0);

    }

    switch_drive_landmark();

    for (tnow = 1; tnow <= tlm_flow; tnow++) {

      if (tscreen && tnow % tscreen == 0)
        printf("lap = %d, flow t = %d\n", lap, tnow); fflush(stdout);
      update_activity(&hist_now, 0);

    }

    for (tnow = 1; tnow <= tlearn; tnow++) {

      if (tscreen && tnow % tscreen == 0)
        printf("lap = %d, learn t = %d\n", lap, tnow); fflush(stdout);
      update_activity(&hist_now, 0);
      update_landmark_counts(lap % 2);

    }

  }

  calculate_landmark_inputs();




  // Main simulation
  printf("\nSTARTING MAIN SIMULATION\n"); fflush(stdout);

  for (lap = 1; lap <= laps_sim; lap++) {

    if (out_psi)
      open_binary(psi_o, "psi", lap);
    if (out_s)
      open_binary(s_o, "s", lap);

    for (tnow = 1; tnow <= tlap; tnow++) {

      if (tscreen && tnow % tscreen == 0)
        printf("lap = %d, t = %d\n", lap, tnow); fflush(stdout);

      update_trajectory(traj_i, &vx, &vy, &state,
                           &vx_old, &vy_old, &state_old, &xpos, &ypos, lap%2);

      update_activity(&hist_now, 0);

      fprintf(traj_o, "%d %6.3f %6.3f %4.3f\n",
                       tnow + tlap*(lap-1), xpos, ypos, state);
      if (out_psi)
        output_potentials_binary(psi_o);
      if (out_s)
        output_spikes_binary(s_o);
     
    }

    if (out_psi)
      close_binary(psi_o);
    if (out_s)
      close_binary(s_o);

  }

  fclose(traj_o);

  
}
