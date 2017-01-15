#include <string.h>
#include "uinput.h"

char path[20] = {0};

void setPath(char *eventX) {
	strcat(path, "/dev/input/");
	strcat(path, eventX);
	printf("%s",path);
}


int goDirection(int pixel) {
struct input_event event, event_end;
  if(!path[0]) {
    printf("Mouse device path is NULL. Call setPath() first.\n");
    return -1;
  }

  int fd = open(path, O_RDWR);
  if (fd < 0) {
    printf("Error open mouse:%s\n", strerror(errno));
    return -1;
  }
  memset(&event, 0, sizeof(event));
  memset(&event, 0, sizeof(event_end));
  gettimeofday(&event.time, NULL);
  event.type = EV_REL;
  event.code = REL_X;
  event.value = pixel;
  gettimeofday(&event_end.time, NULL);
  event_end.type = EV_SYN;
  event_end.code = SYN_REPORT;
  event_end.value = 0;
  write(fd, &event, sizeof(event));// Move the mouse
  write(fd, &event_end, sizeof(event_end));// Show move
  //usleep(1001*20);
  close(fd);
  return pixel;
}

