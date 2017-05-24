#include <assert.h>
#include <fcntl.h>
#include <inttypes.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

#include "atomic.h"
#include "tinycthread.h"

/**
 * Config
 */

#define TIMEOUT        5
#define NUM_FILES      10
#define WRITE_SIZE     100

/**
 * Useful macros
 */

#define __likely(_x)   __builtin_expect((_x), 1)
#define __unlikely(_x) __builtin_expect((_x), 0)
#define min(_a, _b)    (((_a)<(_b))?(_a):(_b))
#define max(_a, _b)    (((_a)>(_b))?(_a):(_b))

/**
 * Type
 */

typedef struct time_stats_t {
  double min_exec;
  double max_exec;
  uint32_t reqs;
} time_stats_t;

static atomic_uint32_t current;
static time_stats_t stats[TIMEOUT];
static time_stats_t result[TIMEOUT];
static int files[NUM_FILES];

/**
 * Thread functions
 */

static
int work(void *arg) {
  char buf[WRITE_SIZE];
  uint32_t current_val = 0;
  double t1, t2;
  while (current_val < TIMEOUT) {

    // Start timer
    t1 = MPI_Wtime();

    // Do something smart here
    int file = files[rand() % NUM_FILES];
    memset(&buf, random(), WRITE_SIZE);
    write(file, &buf, WRITE_SIZE);

    // Stop timer
    t2 = MPI_Wtime();

    // Update stats
    time_stats_t *current_stats = &stats[current_val];
    double elapsed_time = t2 - t1;
    double min_exec = min(current_stats->min_exec, elapsed_time);
    current_stats->min_exec = __unlikely(min_exec == 0.0) ? elapsed_time : min_exec;
    current_stats->max_exec = max(current_stats->max_exec, elapsed_time);
    ++current_stats->reqs;

    // Update val
    current_val = atomic_load_explicit(&current, memory_order_relaxed);

  }
  return 0;
}

static
int update(void *arg) {
  uint32_t val = 0;
  while(__likely(val < TIMEOUT)) {
    sleep(1);
    val = atomic_fetch_add_explicit(&current, 1, memory_order_relaxed);
  }
  return 0;
}

static
void merge_stats(void *inbytes, void *inoutbytes, int *len, MPI_Datatype *dptr) {
  time_stats_t (*in)[TIMEOUT] = inbytes;
  time_stats_t (*inout)[TIMEOUT] = inoutbytes;
  uint32_t sec = 0;
  for (; sec < TIMEOUT; ++sec) {
    inout[sec]->min_exec = min(in[sec]->min_exec, inout[sec]->min_exec);
    inout[sec]->max_exec = max(in[sec]->max_exec, inout[sec]->max_exec);
    inout[sec]->reqs = in[sec]->reqs + inout[sec]->reqs;
  }
}

static
void print_stats() {
  uint32_t sec = 0;
  for (; sec < TIMEOUT; ++sec) {
    printf("%u\t%u\t%f\t%f\n", sec, stats[sec].reqs, stats[sec].min_exec, stats[sec].max_exec);
  }
}

/**
 * Main
 */

int main(int argc, char** argv) {

  // Initialize the MPI environment
  MPI_Init(&argc, &argv);

  // Create MPI types
  MPI_Datatype packedstats;
  MPI_Type_contiguous(sizeof(time_stats_t) * TIMEOUT, MPI_BYTE, &packedstats);
  MPI_Type_commit(&packedstats);

  // Create MPI merge definition
  MPI_Op merge;
  MPI_Op_create(merge_stats, true, &merge);

  // Get the number of processes
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Get the rank of the process
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // Initialize context
  atomic_init(&current, 0);
  memset(stats, 0, sizeof(time_stats_t) * TIMEOUT);
  memset(result, 0, sizeof(time_stats_t) * TIMEOUT);

  // Open files
  uint32_t base_path_len = (uint32_t) strlen(argv[1]);
  uint32_t i = 0;
  for (; i < NUM_FILES; ++i) {
    char path[100];
    memcpy(&path[0], argv[1], base_path_len);
    snprintf(&path[base_path_len], 10, "%" PRIu32, i);
    files[i] = open((char *) &path, O_CREAT | O_WRONLY | O_APPEND, S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
    assert(files[i] != -1);
  }

  // Before starting work
  MPI_Barrier(MPI_COMM_WORLD);

  // Start updater thread
  thrd_t updater;
  thrd_create(&updater, update, NULL);

  // Start worker thread
  thrd_t worker;
  thrd_create(&worker, work, NULL);

  // Join threads
  thrd_join(updater, NULL);
  thrd_join(worker, NULL);

  // Merge time stats
  MPI_Reduce(stats, result, 1, packedstats, merge, 0, MPI_COMM_WORLD);

  // Print results
  if (world_rank == 0) {
    print_stats();
  }

  // Before starting work
  MPI_Barrier(MPI_COMM_WORLD);

  // Finalize the MPI environment.
  MPI_Finalize();

}