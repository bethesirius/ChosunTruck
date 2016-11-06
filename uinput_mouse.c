#include "uinput.h"

int goDirection(int key) {
struct input_event event, event_end;
  int fd = open("/dev/input/mouse0", O_RDWR);
  if (fd < 0) {
    printf("Errro open mouse:%s\n", strerror(errno));
    return -1;
  }
  memset(&event, 0, sizeof(event));
  memset(&event, 0, sizeof(event_end));
  gettimeofday(&event.time, NULL);
  event.type = EV_REL;
  event.code = REL_X;
  event.value = 100;
  gettimeofday(&event_end.time, NULL);
  event_end.type = EV_SYN;
  event_end.code = SYN_REPORT;
  event_end.value = 0;
  write(fd, &event, sizeof(event));// Move the mouse
  write(fd, &event_end, sizeof(event_end));// Show move
  usleep(1001*20);
  close(fd);
  return 0;
}

