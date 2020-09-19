from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import StringProperty
from kivy.config import Config
from kivy.animation import Animation
from kivy.clock import Clock

import numpy as np

import torus_network as tn

Config.set("graphics", "width", "375")
Config.set("graphics", "height", "812")

# ------- Parameters -------
n_h = 256  # Height of a plane where neurons are located
n_w = 256  # Width of a plane where neurons are located
n_connect = 64  # Number of presynaptic neurons a neuron has

proj_ratio = 0.5  # Ratio of projection neurons
sigma_inter = 4  # Standard deviation of distance to other neurons

inhib_ratio = 0.2  # Ratio of interneurons
w_mu = 0.25  # Mean value of weight
w_sigma = 0.08  # Standard deviation of weight

delta_b = 0.01  # change of bias at every time step
mu_u = -0.12
excite_ratio = 0.5
ramda_w = 0.0125   # Hebbian learning ratio

desire_ratio = 0.001
decay_ratio = 0.85
desire_excite_ratio = 1.0

# ------- Initialize network -------
tnet = tn.TorusNetwork(n_h, n_w, n_connect)
tnet.connect(proj_ratio, sigma_inter)
tnet.initialize_network(inhib_ratio, w_mu, w_sigma)
tnet.initialize_desire(desire_ratio)


class MainWidget(Widget):
    image_src = StringProperty()
    ball_src = StringProperty()
    text = StringProperty()

    def __init__(self, **kwargs):
        super(MainWidget, self).__init__(**kwargs)
        self.text = ""
        self.image_src = "./robot.png"
        self.ball_src = "./ball.png"
        self.time = 0
        Clock.schedule_interval(self.update, 0.1)
        self.jump = 0
        self.jump_duration = 0.4

    def update(self, dt):

        tnet.forward(delta_b, mu_u, excite_ratio, ramda_w,
                     decay_ratio, desire_excite_ratio)

        self.time += 1
        if self.time >= 10:
            robot = self.ids.rbt
            anim = Animation(x=robot.x, y=robot.y+self.jump, duration=self.jump_duration) + \
                Animation(x=robot.x, y=robot.y, duration=self.jump_duration)
            anim.start(robot)

            y = tnet.y.reshape(n_h, n_w)

            self.moveArms(y)

            message = ""
            n_sect = 4
            nh = n_h // n_sect
            nw = n_w // n_sect
            for i in range(n_sect):
                for j in range(n_sect):
                    total = np.sum(y[i*nh:(i+1)*nh, j*nw:(j+1)*nw])
                    e_ratio = total / (nh * nw)
                    if e_ratio < 0.5:
                        message += "A"
                    else:
                        message += "P"

            self.text = message

            self.time = 0

    def moveArms(self, y):
        arms = [self.ids.a1, self.ids.a2, self.ids.a3, self.ids.a4]
        n_sect1 = 2
        nh1 = n_h // n_sect1
        nw1 = n_w // n_sect1
        n_sect2 = 2
        nh2 = nh1 // n_sect2
        nw2 = nw1 // n_sect2
        for i in range(n_sect1):
            for j in range(n_sect1):
                ys = y[i * nh1:(i + 1) * nh1, j * nw1:(j + 1) * nw1]
                e_ratio1 = np.sum(ys[0:nh2, 0:nw2]) / (nh2 * nw2)
                e_ratio2 = np.sum(ys[nh2:nh2*2, 0:nw2]) / (nh2 * nw2)
                e_ratio3 = np.sum(ys[0:nh2, nw2:nw2*2]) / (nh2 * nw2)
                e_ratio4 = np.sum(ys[nh2:nh2 * 2, nw2:nw2 * 2]) / (nh2 * nw2)

                d1 = float(2*e_ratio1 - 1)
                d2 = float(2*e_ratio2 - 1)
                d3 = float(0.45*e_ratio3)
                d4 = float(0.45 * e_ratio4)

                dist = 200
                arm = arms[i * n_sect1 + j]
                x_ini = arm.x
                y_ini = arm.y
                anim = Animation(x=arm.x+round(dist*d1), y=arm.y+round(dist*d2), duration=d3) + \
                    Animation(x=x_ini, y=y_ini, duration=d4)
                arm.x = x_ini
                arm.y = y_ini

                anim.start(arm)

    def startPraise(self):
        tnet.emotion = 1
        self.jump = 5
        self.jump_duration = 0.2

    def stopPraise(self):
        tnet.emotion = 0
        self.jump = 0


class GhostApp(App):
    def __init__(self, **kwargs):
        super(GhostApp, self).__init__(**kwargs)
        self.title = "Emo Ghost"


if __name__ == "__main__":

    GhostApp().run()
