#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <linux/input.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <stdlib.h>
#include <linux/uinput.h>

#define die(str, args...) do { \
        perror(str); \
        exit(EXIT_FAILURE); \
    } while(0)

static int fd;//uinput fd
static struct uinput_user_dev uidev;
static struct input_event ev;

int setUinput();
void moveMouse(int);

