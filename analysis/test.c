#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Test typedef
typedef struct {
    int x;
    int y;
} Point;

// Test enum
enum Color {
    RED,
    GREEN,
    BLUE
};

// Test global variable
int global_counter = 0;

// Function declaration (will be in dependencies)
int helper_function(int a, int b);

// Simple function with no dependencies
int add_numbers(int a, int b) {
    return a + b;
}

// Function that uses typedef and global variable
Point create_point(int x, int y) {
    Point p;
    p.x = x;
    p.y = y;
    global_counter++;
    return p;
}

// Function that calls another function
int multiply_and_add(int a, int b, int c) {
    int result = add_numbers(a, b);
    return result * c;
}

// Function that uses enum and standard library
void print_color(enum Color color) {
    switch(color) {
        case RED:
            printf("Red\n");
            break;
        case GREEN:
            printf("Green\n");
            break;
        case BLUE:
            printf("Blue\n");
            break;
        default:
            printf("Unknown color\n");
    }
}

// Function with string manipulation
char* duplicate_string(const char* src) {
    if (src == NULL) {
        return NULL;
    }
    
    size_t len = strlen(src);
    char* dest = malloc(len + 1);
    if (dest == NULL) {
        return NULL;
    }
    
    strcpy(dest, src);
    return dest;
}

// Function that uses Point struct
double calculate_distance(Point p1, Point p2) {
    int dx = p2.x - p1.x;
    int dy = p2.y - p1.y;
    return sqrt(dx * dx + dy * dy);
}

// Helper function implementation
int helper_function(int a, int b) {
    return a - b;
}

// Main function for testing
int main() {
    Point p1 = create_point(0, 0);
    Point p2 = create_point(3, 4);
    
    int sum = add_numbers(10, 20);
    int result = multiply_and_add(5, 10, 2);
    
    print_color(RED);
    
    char* str = duplicate_string("Hello, World!");
    if (str) {
        printf("Duplicated string: %s\n", str);
        free(str);
    }
    
    printf("Sum: %d\n", sum);
    printf("Result: %d\n", result);
    printf("Global counter: %d\n", global_counter);
    
    return 0;
}
