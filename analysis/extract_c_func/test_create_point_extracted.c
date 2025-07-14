/* Extracted from test.c */

/* Includes */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Typedefs */
typedef struct {
    int x;
    int y;
} Point;

/* Enums */
enum Color {
    RED,
    GREEN,
    BLUE
};

/* Structs */
struct {
    int x;
    int y;
};

/* Global Variables */
int global_counter = 0;

/* Function Declarations */
int helper_function(int a, int b);

/* Function Definitions */
Point create_point(int x, int y) {
    Point p;
    p.x = x;
    p.y = y;
    global_counter++;
    return p;
}

