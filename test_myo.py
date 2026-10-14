from myo_raw import MyoRaw

import matplotlib.pyplot as plt
from collections import deque
import threading
import time


# =========================================================
# NASTAVENIA
# =========================================================

MAX_POINTS = 1000
FPS = 60

# Menšie číslo = vizuálne citlivejší graf
SENSITIVITY = 0.15


# =========================================================
# DÁTA
# =========================================================

data = [
    deque(maxlen=MAX_POINTS)
    for _ in range(8)
]

latest = [0] * 8

lock = threading.Lock()

running = True


# =========================================================
# EMG CALLBACK
# =========================================================

def emg_handler(emg, moving):

    with lock:

        for i in range(8):

            data[i].append(emg[i])

            latest[i] = emg[i]


# =========================================================
# MYO
# =========================================================

m = MyoRaw()

m.add_emg_handler(
    emg_handler
)

print()
print("======================================")
print("          MYO EMG MONITOR")
print("======================================")
print()

print("Pripájam Myo...")

m.connect()

print("Myo pripojené.")
print()


# =========================================================
# MYO THREAD
# =========================================================

def myo_loop():

    global running
    global m

    while running:

        try:

            print("Myo: čítam dáta...")

            while running:

                # Toto je pôvodný funkčný spôsob
                # komunikácie s Myo.
                m.run()


        except Exception as e:

            if not running:
                break

            print()
            print("======================================")
            print("MYO ERROR")
            print("======================================")
            print(e)
            print()
            print("Pokus o reconnect o 2 sekundy...")
            print()


            # -----------------------------------------
            # ODPOJ STARÉ MYO
            # -----------------------------------------

            try:

                m.disconnect()

            except:

                pass


            # -----------------------------------------
            # POČKAJ
            # -----------------------------------------

            time.sleep(2)


            if not running:
                break


            # -----------------------------------------
            # RECONNECT
            # -----------------------------------------

            try:

                print("Reconnectujem Myo...")

                new_myo = MyoRaw()

                new_myo.add_emg_handler(
                    emg_handler
                )

                new_myo.connect()

                m = new_myo

                print("Myo znovu pripojené.")
                print()


            except Exception as reconnect_error:

                print()
                print("Reconnect zlyhal:")
                print(reconnect_error)
                print()
                print("Ďalší pokus o 2 sekundy...")
                print()

                time.sleep(2)


# =========================================================
# SPUSTENIE MYO THREADU
# =========================================================

myo_thread = threading.Thread(
    target=myo_loop,
    daemon=True
)

myo_thread.start()


# =========================================================
# GRAF
# =========================================================

plt.style.use("dark_background")

fig, ax = plt.subplots(
    figsize=(15, 8)
)

fig.canvas.manager.set_window_title(
    "MYO EMG Monitor"
)


fig.suptitle(
    "MYO EMG  •  REAL-TIME SIGNAL",
    fontsize=19,
    fontweight="bold",
    y=0.97
)


# =========================================================
# 8 LÍNIÍ
# =========================================================

lines = []

for i in range(8):

    line, = ax.plot(
        [],
        [],
        linewidth=1.5,
        label=f"CH {i + 1}"
    )

    lines.append(line)


# =========================================================
# VZHĽAD GRAFU
# =========================================================

ax.set_xlim(
    0,
    MAX_POINTS
)

ax.set_ylim(
    0,
    9600
)

ax.set_xlabel(
    "Samples",
    fontsize=11
)

ax.set_ylabel(
    "EMG amplitude",
    fontsize=11
)


ax.grid(
    True,
    alpha=0.18,
    linewidth=0.7
)


ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


ax.legend(
    loc="upper left",
    ncol=8,
    frameon=False,
    fontsize=9
)


plt.subplots_adjust(
    left=0.06,
    right=0.985,
    top=0.90,
    bottom=0.08
)


plt.ion()

plt.show(
    block=False
)

plt.pause(0.2)


print("Graf beží.")
print("Zatni sval alebo pohybuj rukou.")
print("Zatvor okno pre ukončenie.")
print()


# =========================================================
# GRAPH LOOP
# =========================================================

frame_time = 1.0 / FPS

next_frame = time.perf_counter()


try:

    while running:

        # ---------------------------------------------
        # SKONTROLUJ OKNO
        # ---------------------------------------------

        if not plt.fignum_exists(fig.number):

            running = False
            break


        # ---------------------------------------------
        # FPS
        # ---------------------------------------------

        now = time.perf_counter()

        if now < next_frame:

            plt.pause(
                0.001
            )

            continue


        next_frame = now + frame_time


        # ---------------------------------------------
        # SKOPÍRUJ DÁTA
        # ---------------------------------------------

        with lock:

            local_data = [
                list(channel)
                for channel in data
            ]

            local_latest = latest.copy()


        # ---------------------------------------------
        # NÁJDI ROZSAH SIGNÁLU
        # ---------------------------------------------

        all_values = []

        for channel in local_data:

            if channel:

                all_values.extend(
                    channel
                )


        if not all_values:

            continue


        minimum = min(
            all_values
        )

        maximum = max(
            all_values
        )


        signal_range = (
            maximum - minimum
        )


        if signal_range < 80:

            signal_range = 80


        # ---------------------------------------------
        # CITLIVOSŤ
        # ---------------------------------------------

        channel_height = (
            signal_range *
            SENSITIVITY
        )


        if channel_height < 150:

            channel_height = 150


        # ---------------------------------------------
        # VYKRESLENIE 8 KANÁLOV
        # ---------------------------------------------

        for i in range(8):

            y = local_data[i]


            if len(y) < 2:

                continue


            # X
            x = range(
                len(y)
            )


            # Každý kanál dostane vlastnú výšku
            offset = (
                (7 - i) *
                channel_height
            )


            # EMG + offset
            plotted_y = [
                value + offset
                for value in y
            ]


            lines[i].set_data(
                x,
                plotted_y
            )


        # ---------------------------------------------
        # Y ROZSAH
        # ---------------------------------------------

        total_height = (
            channel_height * 8
        )


        ax.set_ylim(
            0,
            total_height
        )


        # ---------------------------------------------
        # AKTUÁLNE HODNOTY
        # ---------------------------------------------

        ax.set_title(
            "   ".join(
                f"CH{i + 1}: {local_latest[i]}"
                for i in range(8)
            ),
            fontsize=10,
            pad=8
        )


        # ---------------------------------------------
        # DRAW
        # ---------------------------------------------

        fig.canvas.draw_idle()

        fig.canvas.flush_events()

        plt.pause(
            0.001
        )


# =========================================================
# UKONČENIE
# =========================================================

except KeyboardInterrupt:

    print()
    print("Ukončujem...")


finally:

    running = False


    try:

        m.disconnect()

    except:

        pass


    plt.close(
        "all"
    )


    print()
    print("Myo odpojené.")
    print("Hotovo.")