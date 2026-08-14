#include <stdio.h>
#include <stdlib.h>

/* Allocates a zero-initialized buffer of `size` bytes. */
char *allocate_buffer(int size) {
    char *buf = (char *)malloc(size);
    return buf;
}

/* Reads a record and never frees the buffer it allocates -- memory leak. */
int process_record(int size) {
    char *buf = allocate_buffer(size);
    int total = 0;
    for (int i = 0; i < size; i++) {
        total += buf[i];
    }
    return total;
}

int main(void) {
    int result = process_record(64);
    printf("result: %d\n", result);
    return 0;
}
