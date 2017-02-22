#include <string.h>
#include "uinput.h"

void setEventAndWrite(__u16 type, __u16 code, __s32 value)
{
    ev.type=type;
    ev.code=code;
    ev.value=value;
    if(write(fd, &ev, sizeof(struct input_event)) < 0)
        die("error: write");
}

int setUinput() {
    fd = open("/dev/uinput", O_WRONLY | O_NONBLOCK);
    if(fd < 0) {
        fd=open("/dev/input/uinput",O_WRONLY|O_NONBLOCK);
        if(fd<0){
            die("error: Can't open an uinput file. see #17.(https://github.com/bethesirius/ChosunTruck/issues/17)");
            return -1;
        }
    }
    if(ioctl(fd, UI_SET_EVBIT, EV_KEY) < 0) {
        die("error: ioctl");
        return -1;
    }
    if(ioctl(fd, UI_SET_KEYBIT, BTN_LEFT) < 0) {
        die("error: ioctl");
        return -1;
    }
    if(ioctl(fd, UI_SET_KEYBIT, KEY_TAB) < 0) {
        die("error: ioctl");
        return -1;
    }
    if(ioctl(fd, UI_SET_KEYBIT, KEY_ENTER) < 0) {
        die("error: ioctl");
        return -1;
    }
    if(ioctl(fd, UI_SET_KEYBIT, KEY_LEFTSHIFT) < 0) {
        die("error: ioctl");
        return -1;
    }
    if(ioctl(fd, UI_SET_EVBIT, EV_REL) < 0) {
        die("error: ioctl");
        return -1;
    }
    if(ioctl(fd, UI_SET_RELBIT, REL_X) < 0) {
        die("error: ioctl");
        return -1;
    }
    if(ioctl(fd, UI_SET_RELBIT, REL_Y) < 0) {
        die("error: ioctl");
        return -1;
    }

    memset(&uidev, 0, sizeof(uidev));
    snprintf(uidev.name, UINPUT_MAX_NAME_SIZE, "uinput-virtualMouse");
    uidev.id.bustype = BUS_USB;
    uidev.id.vendor  = 0x1;
    uidev.id.product = 0x1;
    uidev.id.version = 1;

    if(write(fd, &uidev, sizeof(uidev)) < 0) {
        die("error: write");
        return -1;
    }
    if(ioctl(fd, UI_DEV_CREATE) < 0) {
        die("error: ioctl");
        return -1;
    }

    memset(&ev, 0, sizeof(struct input_event));
    setEventAndWrite(EV_REL,REL_X,1);
    setEventAndWrite(EV_REL,REL_Y,1);
    setEventAndWrite(EV_SYN,0,0);
    return 0;
}

void moveMouse(int x) {
    int dx=x;
    setEventAndWrite(EV_REL,REL_X,dx);
    setEventAndWrite(EV_SYN,0,0);
}


