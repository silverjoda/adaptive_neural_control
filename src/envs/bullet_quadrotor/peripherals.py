import logging
import pygame

class JoyController():
    def __init__(self, config):
        self.config = config
        logging.info("Initializing joystick controller")
        pygame.init()
        if self.config["target_input_source"] == "joystick":
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            logging.info("Initialized gamepad: {}".format(self.joystick.get_name()))
        else:
            logging.info("No joystick found")
        logging.info("Finished initializing the joystick controller.")
        self.button_x_state = 0

    def get_joystick_input(self):
        pygame.event.pump()
        t_roll, t_pitch, t_yaw, throttle = \
            [self.joystick.get_axis(self.config["joystick_mapping"][i]) for i in range(4)]
        button_x = self.joystick.get_button(1)
        pygame.event.clear()

        # button_x only when upon press
        if self.button_x_state == 0 and button_x == 1:
            self.button_x_state = 1
            button_x_event = 1
        elif self.button_x_state == 1 and button_x == 0:
            self.button_x_state = 0
            button_x_event = 0
        elif self.button_x_state == 1 and button_x == 1:
            self.button_x_state = 1
            button_x_event = 0
        else:
            self.button_x_state = 0
            button_x_event = 0

        return throttle, -t_roll, t_pitch, t_yaw, button_x, button_x_event
