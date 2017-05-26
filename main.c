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

typedef struct context_t {
  uint32_t timeout;
  uint32_t num_files;
  uint32_t write_size;
  atomic_uint32_t current;
  time_stats_t *stats;
  time_stats_t *result;
  int *files;
} context_t;


/**
 * Thread functions
 */

static
int work(void *arg) {
  context_t *context = (context_t *) arg;
  char buf[context->write_size];
  uint32_t current_val = 0;
  double t1, t2;
  while (current_val < context->timeout) {

    // Start timer
    t1 = MPI_Wtime();

    // Do something smart here
    int file = context->files[rand() % context->num_files];
    memset(&buf, random(), context->write_size);
    write(file, &buf, context->write_size);

    // Stop timer
    t2 = MPI_Wtime();

    // Update stats
    time_stats_t *current_stats = &context->stats[current_val];
    double elapsed_time = t2 - t1;
    double min_exec = min(current_stats->min_exec, elapsed_time);
    current_stats->min_exec = __unlikely(min_exec == 0.0) ? elapsed_time : min_exec;
    current_stats->max_exec = max(current_stats->max_exec, elapsed_time);
    ++current_stats->reqs;

    // Update val
    current_val = atomic_load_explicit(&context->current, memory_order_relaxed);

  }
  return 0;
}

static
int update(void *arg) {
  context_t *context = (context_t *) arg;
  uint32_t val = 0;
  while(__likely(val < context->timeout)) {
    sleep(1);
    val = atomic_fetch_add_explicit(&context->current, 1, memory_order_relaxed);
  }
  return 0;
}

static
void merge_stats(time_stats_t *in, time_stats_t *inout, int *len, MPI_Datatype *dptr) {
  int i = 0;
  for (; i < *len; ++i) {
    (inout + i)->min_exec = min((in + i)->min_exec, (inout + i)->min_exec);
    (inout + i)->max_exec = max((in + i)->max_exec, (inout + i)->max_exec);
    (inout + i)->reqs = (in + i)->reqs + (inout + i)->reqs;
  }
}

static
void print_stats(context_t *context) {
  uint32_t sec = 0;
  for (; sec < context->timeout; ++sec) {
    printf("%u\t%u\t%f\t%f\n", sec, context->result[sec].reqs, context->result[sec].min_exec, context->result[sec].max_exec);
  }
}

/**
 * Main
 */

int main(int argc, char** argv) {

  // Check arguments
  if (argc != 5) {
    printf("Usage: %s file_template timeout write_size n_files\n", argv[0]);
    exit(1);
  }

  // Initialize the MPI environment
  MPI_Init(&argc, &argv);

  // Create context
  uint32_t timeout = (uint32_t) atoi(argv[2]);
  uint32_t write_size = (uint32_t) atoi(argv[3]);
  uint32_t num_files = (uint32_t) atoi(argv[4]);
  time_stats_t stats[timeout];
  time_stats_t result[timeout];
  int files[num_files];
  context_t context = {
      .current = ATOMIC_VAR_INIT(0),
      .stats = &stats[0],
      .result = &result[0],
      .files = &files[0],
      .timeout = timeout,
      .write_size = write_size,
      .num_files = num_files
  };

  // Create MPI types
  MPI_Datatype packedstat;
  MPI_Type_contiguous(sizeof(time_stats_t), MPI_BYTE, &packedstat);
  MPI_Type_commit(&packedstat);

  // Create MPI merge definition
  MPI_Op merge;
  MPI_Op_create((MPI_User_function *) merge_stats, true, &merge);

  // Get the number of processes
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Get the rank of the process
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // Initialize context
  memset(stats, 0, sizeof(time_stats_t) * timeout);
  memset(result, 0, sizeof(time_stats_t) * timeout);

  // Open files
  uint32_t base_path_len = (uint32_t) strlen(argv[1]);
  uint32_t i = 0;
  for (; i < num_files; ++i) {
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
  thrd_create(&updater, update, &context);

  // Start worker thread
  thrd_t worker;
  thrd_create(&worker, work, &context);

  // Join threads
  thrd_join(updater, NULL);
  thrd_join(worker, NULL);

  // Merge time stats
  MPI_Reduce(&stats[0], &result[0], timeout, packedstat, merge, 0, MPI_COMM_WORLD);

  // Print results
  if (world_rank == 0) {
    print_stats(&context);
  }

  // Before starting work
  MPI_Barrier(MPI_COMM_WORLD);

  // Finalize the MPI environment.
  MPI_Finalize();

}