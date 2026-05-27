"""
╔══════════════════════════════════════════════════════════╗
║         ROCKET TRAJECTORY SIMULATOR  — Python            ║
║   Multi-Stage · Drag · Planets · Escape Velocity         ║
║                                                          ║
║  Requirements:  pip install matplotlib numpy             ║
║  Run:           python rocket_simulator.py               ║
╚══════════════════════════════════════════════════════════╝
"""

import tkinter as tk
from tkinter import ttk, font as tkfont
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────────────────
#  PLANET DATA
# ─────────────────────────────────────────────────────────
PLANETS = {
    "Earth":   {"g": 9.81,  "rho0": 1.225, "H": 8500,  "esc_v": 11200, "color": "#4488ff"},
    "Moon":    {"g": 1.62,  "rho0": 0.0,   "H": 1,     "esc_v": 2380,  "color": "#aaaaaa"},
    "Mars":    {"g": 3.72,  "rho0": 0.020, "H": 11100, "esc_v": 5030,  "color": "#ff6633"},
    "Venus":   {"g": 8.87,  "rho0": 65.0,  "H": 15900, "esc_v": 10360, "color": "#ffcc44"},
    "Jupiter": {"g": 24.8,  "rho0": 1.3,   "H": 27000, "esc_v": 59500, "color": "#ee8833"},
    "Titan":   {"g": 1.35,  "rho0": 5.3,   "H": 40000, "esc_v": 2640,  "color": "#ccaa77"},
}

# ─────────────────────────────────────────────────────────
#  SIMULATION ENGINE
# ─────────────────────────────────────────────────────────
def run_simulation(params):
    """
    Numerically integrates rocket equations of motion.

    Physics:
      F_net  = Thrust − Weight − Drag
      a      = F_net / m(t)           (Newton's 2nd law)
      v(t+dt)= v(t) + a·dt
      h(t+dt)= h(t) + v·dt

    Drag (if enabled):
      F_drag = 0.5 · ρ(h) · Cd · A · v²
      ρ(h)   = ρ₀ · exp(−h / H)      (barometric formula)

    Mass decreases during burn:
      ṁ = fuel_mass / burn_time  (constant burn-rate assumption)
    """
    p          = params
    planet     = PLANETS[p["planet"]]
    g          = planet["g"]
    rho0       = planet["rho0"]
    H_scale    = planet["H"]

    payload    = p["payload"]
    dry_mass   = p["dry_mass"]
    fuel1      = p["fuel1"]
    thrust1    = p["thrust1"] * 1000          # kN → N
    burn1      = max(p["burn1"], 1.0)
    burn_rate1 = fuel1 / burn1

    two_stage  = p["two_stage"]
    fuel2      = p["fuel2"] if two_stage else 0.0
    thrust2    = p["thrust2"] * 1000 if two_stage else 0.0
    burn_rate2 = (fuel2 / max(fuel2 / burn_rate1 * 0.8, 1)) if (two_stage and fuel2 > 0) else 0

    drag_on    = p["drag"]
    Cd         = 0.3
    A          = 10.0                          # cross-section m²

    dt         = 0.5                           # timestep (seconds)
    t, v, h    = 0.0, 0.0, 0.0
    f1, f2     = fuel1, fuel2
    stage      = 1
    stage2_t   = None

    times, alts, vels, accs = [], [], [], []

    while t < 3000 and h >= -1:
        # ── current total mass ──
        m = dry_mass + payload + f1 + f2

        # ── thrust & fuel burn ──
        thrust_now = 0.0
        if stage == 1 and f1 > 0:
            thrust_now = thrust1
            f1 = max(0.0, f1 - burn_rate1 * dt)
            if f1 == 0.0 and two_stage:
                stage = 2
                stage2_t = t
        elif stage == 2 and f2 > 0:
            thrust_now = thrust2
            f2 = max(0.0, f2 - burn_rate2 * dt)

        # ── drag force ──
        drag_force = 0.0
        if drag_on and h >= 0:
            rho = rho0 * np.exp(-h / H_scale)
            drag_force = 0.5 * rho * Cd * A * v * abs(v)

        # ── net force & acceleration ──
        weight  = m * g
        F_net   = thrust_now - weight - drag_force
        acc     = F_net / m

        times.append(round(t, 2))
        alts.append(h / 1000.0)               # m → km
        vels.append(v)
        accs.append(acc)

        # ── integrate ──
        v += acc * dt
        h += v   * dt
        t += dt

        if h < 0 and v < 0:
            break

    results = {
        "times":      np.array(times),
        "alts":       np.array(alts),
        "vels":       np.array(vels),
        "accs":       np.array(accs),
        "max_alt":    max(alts) if alts else 0,
        "max_vel":    max(abs(v) for v in vels) if vels else 0,
        "flight_time":times[-1] if times else 0,
        "twr":        thrust1 / ((payload + fuel1 + fuel2 + dry_mass) * g),
        "stage2_t":   stage2_t,
        "planet":     p["planet"],
        "esc_v":      planet["esc_v"],
    }
    return results


# ─────────────────────────────────────────────────────────
#  DARK THEME HELPERS
# ─────────────────────────────────────────────────────────
BG       = "#0a0a0f"
PANEL_BG = "#0d1117"
ACCENT   = "#00ff88"
DIM      = "#4a7a4a"
BORDER   = "#1e2a1e"
TEXT     = "#e0e0e0"
BLUE     = "#44aaff"
ORANGE   = "#ff9944"
RED      = "#ff4444"

PLOT_BG  = "#060e08"
GRID_CLR = "#0f1f0f"
TICK_CLR = "#4a7a4a"

def style_label(parent, text, size=10, color=TEXT, anchor="w", **kw):
    return tk.Label(parent, text=text, bg=PANEL_BG, fg=color,
                    font=("Courier New", size), anchor=anchor, **kw)

def style_slider(parent, from_, to, resolution, variable, command=None):
    s = tk.Scale(parent, from_=from_, to=to, resolution=resolution,
                 orient="horizontal", variable=variable, command=command,
                 bg=PANEL_BG, fg=ACCENT, troughcolor=BORDER,
                 activebackground=ACCENT, highlightthickness=0,
                 sliderrelief="flat", length=240, showvalue=False)
    return s

def section_label(parent, text):
    f = tk.Frame(parent, bg=BORDER, height=1)
    f.pack(fill="x", pady=(10, 4))
    lbl = tk.Label(parent, text=f"▸ {text}", bg=PANEL_BG, fg=ACCENT,
                   font=("Courier New", 9, "bold"), anchor="w")
    lbl.pack(fill="x", padx=4, pady=(0, 4))


# ─────────────────────────────────────────────────────────
#  MAIN APPLICATION
# ─────────────────────────────────────────────────────────
class RocketSimApp:
    def __init__(self, root):
        self.root = root
        self.root.title("⟨ Rocket Trajectory Simulator ⟩")
        self.root.configure(bg=BG)
        self.root.geometry("1280x860")
        self.root.minsize(1000, 700)

        # ── Tkinter variables ──
        self.planet_var   = tk.StringVar(value="Earth")
        self.payload_var  = tk.DoubleVar(value=500)
        self.fuel1_var    = tk.DoubleVar(value=10000)
        self.dry_var      = tk.DoubleVar(value=2000)
        self.thrust1_var  = tk.DoubleVar(value=500)
        self.burn1_var    = tk.DoubleVar(value=60)
        self.fuel2_var    = tk.DoubleVar(value=3000)
        self.thrust2_var  = tk.DoubleVar(value=200)
        self.drag_var     = tk.BooleanVar(value=False)
        self.stage2_var   = tk.BooleanVar(value=True)

        self._build_ui()

    # ──────────────────────────────────────────────────
    #  UI CONSTRUCTION
    # ──────────────────────────────────────────────────
    def _build_ui(self):
        # ── Header ──
        hdr = tk.Frame(self.root, bg=BG, pady=10)
        hdr.pack(fill="x")
        tk.Label(hdr, text="⟨ ROCKET TRAJECTORY SIMULATOR ⟩",
                 bg=BG, fg=ACCENT, font=("Courier New", 16, "bold")).pack()
        tk.Label(hdr, text="Multi-Stage  ·  Drag Analysis  ·  Planet Comparison  ·  Escape Velocity",
                 bg=BG, fg=DIM, font=("Courier New", 9)).pack()

        sep = tk.Frame(self.root, bg=BORDER, height=1)
        sep.pack(fill="x", padx=10)

        # ── Main layout: left panel + right charts ──
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill="both", expand=True, padx=10, pady=10)

        left = tk.Frame(main, bg=PANEL_BG, bd=0,
                        highlightbackground=BORDER, highlightthickness=1)
        left.pack(side="left", fill="y", padx=(0, 10))
        left.pack_propagate(False)
        left.configure(width=310)

        right = tk.Frame(main, bg=BG)
        right.pack(side="left", fill="both", expand=True)

        self._build_left_panel(left)
        self._build_right_panel(right)

    def _build_left_panel(self, parent):
        canvas = tk.Canvas(parent, bg=PANEL_BG, highlightthickness=0)
        scrollbar = tk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scroll_frame = tk.Frame(canvas, bg=PANEL_BG)

        scroll_frame.bind("<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        inner = scroll_frame
        pad = {"padx": 10, "pady": 2}

        # ── Planet selector ──
        section_label(inner, "Target Planet")
        pg = tk.Frame(inner, bg=PANEL_BG)
        pg.pack(fill="x", padx=10, pady=4)
        for i, name in enumerate(PLANETS):
            col = PLANETS[name]["color"]
            btn = tk.Button(pg, text=name,
                            font=("Courier New", 8),
                            bg="#0a1a0e", fg=DIM,
                            activebackground=col, activeforeground=BG,
                            relief="flat", bd=1,
                            padx=4, pady=3,
                            command=lambda n=name: self._select_planet(n))
            btn.grid(row=i//3, column=i%3, padx=2, pady=2, sticky="ew")
            pg.columnconfigure(i%3, weight=1)
            setattr(self, f"btn_{name}", btn)
        self._select_planet("Earth")

        # ── Rocket config sliders ──
        section_label(inner, "Rocket Config")

        sliders1 = [
            ("Payload Mass (kg)",  "payload_var",  100,  5000,   100),
            ("Fuel Mass — S1 (kg)","fuel1_var",   1000, 100000, 1000),
            ("Dry Mass (kg)",      "dry_var",      500,  20000,   500),
            ("Thrust S1 (kN)",     "thrust1_var",   50,   5000,    50),
            ("Burn Time S1 (s)",   "burn1_var",     10,    600,     5),
        ]
        for lbl_text, var_name, mn, mx, res in sliders1:
            self._make_slider_row(inner, lbl_text, var_name, mn, mx, res)

        # ── Options ──
        section_label(inner, "Options")
        opt_f = tk.Frame(inner, bg=PANEL_BG)
        opt_f.pack(fill="x", padx=10, pady=4)

        self.drag_btn = tk.Checkbutton(opt_f, text=" Air Resistance (Drag)",
                                       variable=self.drag_var,
                                       bg=PANEL_BG, fg=TEXT,
                                       selectcolor=BG, activebackground=PANEL_BG,
                                       font=("Courier New", 9),
                                       command=self._on_toggle)
        self.drag_btn.pack(anchor="w")

        self.stage2_btn = tk.Checkbutton(opt_f, text=" 2-Stage Rocket",
                                         variable=self.stage2_var,
                                         bg=PANEL_BG, fg=TEXT,
                                         selectcolor=BG, activebackground=PANEL_BG,
                                         font=("Courier New", 9),
                                         command=self._on_toggle)
        self.stage2_btn.pack(anchor="w")

        # ── Stage 2 config ──
        self.s2_frame = tk.Frame(inner, bg=PANEL_BG)
        self.s2_frame.pack(fill="x")
        section_label(self.s2_frame, "Stage 2 Config")
        sliders2 = [
            ("Fuel Mass — S2 (kg)", "fuel2_var",    500, 30000, 500),
            ("Thrust S2 (kN)",      "thrust2_var",   50,  2000,  50),
        ]
        for lbl_text, var_name, mn, mx, res in sliders2:
            self._make_slider_row(self.s2_frame, lbl_text, var_name, mn, mx, res)

        # ── Launch button ──
        tk.Frame(inner, bg=PANEL_BG, height=8).pack()
        launch = tk.Button(inner, text="⚡  LAUNCH SIMULATION",
                           font=("Courier New", 11, "bold"),
                           bg="#001a0d", fg=ACCENT,
                           activebackground=ACCENT, activeforeground=BG,
                           relief="flat", bd=1, pady=10,
                           command=self._launch)
        launch.pack(fill="x", padx=10, pady=(4, 12))

    def _make_slider_row(self, parent, label_text, var_name, mn, mx, res):
        row = tk.Frame(parent, bg=PANEL_BG)
        row.pack(fill="x", padx=10, pady=1)
        top = tk.Frame(row, bg=PANEL_BG)
        top.pack(fill="x")
        tk.Label(top, text=label_text, bg=PANEL_BG, fg=DIM,
                 font=("Courier New", 8), anchor="w").pack(side="left")
        val_lbl = tk.Label(top, text=str(int(getattr(self, var_name).get())),
                           bg=PANEL_BG, fg=ACCENT,
                           font=("Courier New", 9, "bold"))
        val_lbl.pack(side="right")

        def on_change(v, lbl=val_lbl, vn=var_name):
            lbl.config(text=f"{int(float(v)):,}")

        s = tk.Scale(row, from_=mn, to=mx, resolution=res,
                     orient="horizontal", variable=getattr(self, var_name),
                     command=on_change,
                     bg=PANEL_BG, fg=ACCENT, troughcolor=BORDER,
                     activebackground=ACCENT, highlightthickness=0,
                     sliderrelief="flat", showvalue=False)
        s.pack(fill="x")
        setattr(self, f"val_{var_name}", val_lbl)

    def _build_right_panel(self, parent):
        # ── Stats row ──
        self.stats_frame = tk.Frame(parent, bg=PANEL_BG,
                                    highlightbackground=BORDER, highlightthickness=1)
        self.stats_frame.pack(fill="x", pady=(0, 8))
        self.stat_labels = {}
        for key, title in [("max_alt","Max Altitude (km)"),
                            ("max_vel","Max Velocity (km/s)"),
                            ("flight","Flight Time (s)"),
                            ("twr","Thrust/Weight"),
                            ("esc","Escape Vel Progress"),
                            ("stage2","Stage 2 Separation")]:
            cell = tk.Frame(self.stats_frame, bg=PANEL_BG, padx=12, pady=8)
            cell.pack(side="left", expand=True, fill="both")
            tk.Label(cell, text=title, bg=PANEL_BG, fg=DIM,
                     font=("Courier New", 7), anchor="w").pack(anchor="w")
            lbl = tk.Label(cell, text="—", bg=PANEL_BG, fg=ACCENT,
                           font=("Courier New", 14, "bold"), anchor="w")
            lbl.pack(anchor="w")
            self.stat_labels[key] = lbl

        # ── Charts (matplotlib embedded) ──
        self.fig = Figure(figsize=(9, 5.5), facecolor=BG)
        self.fig.subplots_adjust(hspace=0.55, left=0.08, right=0.97,
                                  top=0.93, bottom=0.10)

        self.ax_alt = self.fig.add_subplot(2, 1, 1)
        self.ax_vel = self.fig.add_subplot(2, 1, 2)

        for ax, title, ylabel in [
            (self.ax_alt, "ALTITUDE vs TIME", "Altitude (km)"),
            (self.ax_vel, "VELOCITY vs TIME",  "Velocity (m/s)")
        ]:
            ax.set_facecolor(PLOT_BG)
            ax.set_title(title, color=ACCENT, fontsize=8,
                         fontfamily="monospace", loc="left", pad=6)
            ax.set_xlabel("Time (s)", color=TICK_CLR, fontsize=8, fontfamily="monospace")
            ax.set_ylabel(ylabel,     color=TICK_CLR, fontsize=8, fontfamily="monospace")
            ax.tick_params(colors=TICK_CLR, labelsize=7)
            ax.grid(color=GRID_CLR, linewidth=0.5, linestyle="--")
            for spine in ax.spines.values():
                spine.set_edgecolor(BORDER)
            ax.text(0.5, 0.5, "Configure & Launch →",
                    transform=ax.transAxes, ha="center", va="center",
                    color="#2a4a2a", fontsize=10, fontfamily="monospace",
                    style="italic")

        self.canvas_widget = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas_widget.get_tk_widget().pack(fill="both", expand=True)
        self.canvas_widget.draw()

    # ──────────────────────────────────────────────────
    #  EVENT HANDLERS
    # ──────────────────────────────────────────────────
    def _select_planet(self, name):
        self.planet_var.set(name)
        for n in PLANETS:
            btn = getattr(self, f"btn_{n}", None)
            if btn:
                if n == name:
                    btn.config(bg="#0a2a18", fg=ACCENT,
                               highlightbackground=ACCENT, highlightthickness=1)
                else:
                    btn.config(bg="#0a1a0e", fg=DIM,
                               highlightbackground=BORDER, highlightthickness=0)

    def _on_toggle(self):
        show = self.stage2_var.get()
        if show:
            self.s2_frame.pack(fill="x")
        else:
            self.s2_frame.pack_forget()

    def _launch(self):
        params = {
            "planet":    self.planet_var.get(),
            "payload":   self.payload_var.get(),
            "fuel1":     self.fuel1_var.get(),
            "dry_mass":  self.dry_var.get(),
            "thrust1":   self.thrust1_var.get(),
            "burn1":     self.burn1_var.get(),
            "fuel2":     self.fuel2_var.get(),
            "thrust2":   self.thrust2_var.get(),
            "drag":      self.drag_var.get(),
            "two_stage": self.stage2_var.get(),
        }
        res = run_simulation(params)
        self._update_stats(res)
        self._update_charts(res)

    # ──────────────────────────────────────────────────
    #  RESULTS
    # ──────────────────────────────────────────────────
    def _update_stats(self, res):
        esc_pct = min(100, (res["max_vel"] / res["esc_v"]) * 100)
        escaped = res["max_vel"] >= res["esc_v"]
        esc_color = RED if escaped else ACCENT
        esc_text = "✓ ESCAPED!" if escaped else f"{esc_pct:.1f}%"

        self.stat_labels["max_alt"].config(text=f"{res['max_alt']:.1f}")
        self.stat_labels["max_vel"].config(text=f"{res['max_vel']/1000:.2f}")
        self.stat_labels["flight"].config(text=f"{res['flight_time']:.0f}")
        self.stat_labels["twr"].config(text=f"{res['twr']:.2f}")
        self.stat_labels["esc"].config(text=esc_text, fg=esc_color)
        s2t = res["stage2_t"]
        self.stat_labels["stage2"].config(
            text=f"T+{s2t:.0f}s" if s2t is not None else "N/A"
        )

    def _update_charts(self, res):
        times = res["times"]
        alts  = res["alts"]
        vels  = res["vels"]
        s2t   = res["stage2_t"]
        pname = res["planet"]
        pcol  = PLANETS[pname]["color"]

        # downsample for performance
        step = max(1, len(times) // 400)
        t  = times[::step]
        h  = alts[::step]
        v  = vels[::step]

        for ax in (self.ax_alt, self.ax_vel):
            ax.cla()
            ax.set_facecolor(PLOT_BG)
            ax.tick_params(colors=TICK_CLR, labelsize=7)
            ax.grid(color=GRID_CLR, linewidth=0.5, linestyle="--")
            for spine in ax.spines.values():
                spine.set_edgecolor(BORDER)

        # altitude plot
        self.ax_alt.plot(t, h, color=ACCENT, linewidth=1.4, label="Altitude")
        self.ax_alt.fill_between(t, h, alpha=0.07, color=ACCENT)
        self.ax_alt.set_title(
            f"ALTITUDE vs TIME  —  {pname}  |  Max: {res['max_alt']:.1f} km",
            color=ACCENT, fontsize=8, fontfamily="monospace", loc="left", pad=6)
        self.ax_alt.set_xlabel("Time (s)", color=TICK_CLR, fontsize=8, fontfamily="monospace")
        self.ax_alt.set_ylabel("Altitude (km)", color=TICK_CLR, fontsize=8, fontfamily="monospace")

        # velocity plot
        self.ax_vel.plot(t, v, color=BLUE, linewidth=1.4, label="Velocity")
        self.ax_vel.fill_between(t, v, alpha=0.07, color=BLUE)
        # escape velocity line
        esc_kms = res["esc_v"]
        if max(abs(vv) for vv in vels) > 0:
            self.ax_vel.axhline(y=esc_kms, color=RED, linewidth=0.8,
                                linestyle="--", alpha=0.7, label=f"Escape vel ({esc_kms/1000:.1f} km/s)")
        self.ax_vel.set_title(
            f"VELOCITY vs TIME  —  Max: {res['max_vel']/1000:.2f} km/s  |  Escape: {esc_kms/1000:.2f} km/s",
            color=BLUE, fontsize=8, fontfamily="monospace", loc="left", pad=6)
        self.ax_vel.set_xlabel("Time (s)", color=TICK_CLR, fontsize=8, fontfamily="monospace")
        self.ax_vel.set_ylabel("Velocity (m/s)", color=TICK_CLR, fontsize=8, fontfamily="monospace")
        self.ax_vel.legend(fontsize=7, facecolor=PLOT_BG, edgecolor=BORDER,
                           labelcolor=TEXT)

        # stage separation markers
        if s2t is not None:
            for ax in (self.ax_alt, self.ax_vel):
                ax.axvline(x=s2t, color=ORANGE, linewidth=1,
                           linestyle="--", alpha=0.8)
                ax.text(s2t + 1, ax.get_ylim()[1] * 0.02,
                        f"Stage Sep\nT+{s2t:.0f}s",
                        color=ORANGE, fontsize=6, fontfamily="monospace",
                        va="bottom")

        self.fig.canvas.draw_idle()


# ─────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()

    # ── dark title bar on Windows ──
    try:
        root.tk.call("source", "azure.tcl")
    except Exception:
        pass
    try:
        import ctypes
        HWND = ctypes.windll.user32.GetForegroundWindow()
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            HWND, 20, ctypes.byref(ctypes.c_int(1)), ctypes.sizeof(ctypes.c_int))
    except Exception:
        pass

    app = RocketSimApp(root)
    root.mainloop()
