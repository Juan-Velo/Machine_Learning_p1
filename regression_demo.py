from manim import *
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.optimize import curve_fit

class MasterRegressionDemo(ThreeDScene):
    def construct(self):
        # ---------------------------------------------------------
        # ESCENA 0: INTRODUCCIÓN
        # ---------------------------------------------------------
        title = Text("Regresión: De Simple a Avanzada", font_size=42, color=BLUE)
        subtitle = Text("Grupo 11 - Machine Learning", font_size=28, color=GRAY).next_to(title, DOWN)
        
        # Iniciar en vista 2D (phi=0, theta=-90 es plano 2D normal)
        self.set_camera_orientation(phi=0, theta=-90 * DEGREES)
        
        self.play(Write(title), FadeIn(subtitle))
        self.wait(3)
        self.play(FadeOut(title), FadeOut(subtitle))

        # ---------------------------------------------------------
        # ESCENA 1: REGRESIÓN LINEAL SIMPLE (2D)
        # ---------------------------------------------------------
        # Configurar ejes 2D
        axes = Axes(
            x_range=[0, 10, 1], y_range=[0, 10, 1],
            x_length=9, y_length=6,
            axis_config={"include_numbers": True}
        ).to_edge(DOWN)
        
        # --- CORRECCIÓN AQUÍ: Usamos \\text{} para la ñ ---
        labels = axes.get_axis_labels(x_label="\\text{Tamaño }(x)", y_label="\\text{Precio }(y)")
        
        self.play(Create(axes), Write(labels))
        
        # Datos Lineales
        np.random.seed(42)
        x_lin = np.linspace(1, 9, 15).reshape(-1, 1)
        y_lin = 0.7 * x_lin + 1.5 + np.random.normal(0, 0.6, size=x_lin.shape)
        
        dots_lin = VGroup()
        for x, y in zip(x_lin.flatten(), y_lin.flatten()):
            dots_lin.add(Dot(point=axes.c2p(x, y), color=YELLOW, radius=0.08))
            
        self.play(ShowIncreasingSubsets(dots_lin), run_time=2)
        
        # Ajuste Lineal
        model_lin = LinearRegression().fit(x_lin, y_lin)
        def pred_lin(x): return model_lin.predict([[x]]).item()
        
        line_lin = axes.plot(pred_lin, color=BLUE, x_range=[0, 10])
        eq_lin = MathTex(r"y = mx + b", color=BLUE).to_edge(UR).shift(LEFT)
        
        self.play(Create(line_lin), Write(eq_lin))
        
        # Mostrar Error (Residuos)
        error_lines = VGroup()
        for x, y in zip(x_lin.flatten(), y_lin.flatten()):
            error_lines.add(Line(axes.c2p(x, y), axes.c2p(x, pred_lin(x)), color=ORANGE))
        
        self.play(Create(error_lines))
        self.wait(2)
        
        # Limpiar Escena 1
        self.play(FadeOut(dots_lin), FadeOut(line_lin), FadeOut(eq_lin), FadeOut(error_lines))

        # ---------------------------------------------------------
        # ESCENA 2: NO LINEAL (EXPONENCIAL)
        # ---------------------------------------------------------
        # Datos Exponenciales
        x_exp = np.linspace(0.5, 7.5, 15)
        y_exp = 0.5 * np.exp(0.5 * x_exp) + 0.5 + np.random.normal(0, 0.5, size=x_exp.shape)
        
        dots_exp = VGroup()
        for x, y in zip(x_exp, y_exp):
            if y < 10: dots_exp.add(Dot(point=axes.c2p(x, y), color=GREEN))
            
        self.play(FadeIn(dots_exp))
        
        # Intento fallido lineal
        model_bad = LinearRegression().fit(x_exp.reshape(-1, 1), y_exp)
        def pred_bad(x): return model_bad.predict([[x]]).item()
        line_bad = axes.plot(pred_bad, color=RED, x_range=[0, 8])
        
        self.play(Create(line_bad))
        self.wait(1)
        
        # Solución No Lineal (Curve Fit)
        def func_exp(x, a, b, c): return a * np.exp(b * x) + c
        popt, _ = curve_fit(func_exp, x_exp, y_exp, p0=[0.5, 0.5, 1])
        def pred_curve(x): return func_exp(x, *popt)
        
        curve_good = axes.plot(pred_curve, color=GREEN, x_range=[0, 8])
        
        self.play(Transform(line_bad, curve_good))
        self.wait(2)
        
        # Limpiar para 3D
        self.play(FadeOut(axes), FadeOut(labels), FadeOut(dots_exp), FadeOut(line_bad))

        # ---------------------------------------------------------
        # ESCENA 3: REGRESIÓN MÚLTIPLE (3D) - CORREGIDA
        # ---------------------------------------------------------
        title_3d = Text("Regresión Múltiple (3D)", font_size=32).to_corner(UL)
        self.add_fixed_in_frame_mobjects(title_3d)
        self.play(Write(title_3d))
        
        # 1. Creamos los ejes
        axes_3d = ThreeDAxes(
            x_range=[0,6], y_range=[0,6], z_range=[0,6], 
            x_length=5, y_length=5, z_length=4
        )
        labels_3d = axes_3d.get_axis_labels(x_label="x1", y_label="x2", z_label="y")
        
        # 2. AJUSTE CLAVE: Movemos los ejes ABAJO e IZQUIERDA.
        # Esto hace que los datos (que son positivos) queden visualmente al centro.
        group_axes = VGroup(axes_3d, labels_3d)
        group_axes.shift(DOWN * 1.5 + LEFT * 1) 

        self.play(Create(axes_3d), Write(labels_3d))
        
        # 3. Mover cámara con ZOOM OUT (0.6) para que nada se salga
        self.move_camera(phi=75 * DEGREES, theta=30 * DEGREES, zoom=0.6, run_time=2)
        
        # Puntos 3D
        x3 = np.random.uniform(1, 5, 15)
        y3 = np.random.uniform(1, 5, 15)
        z3 = 0.5*x3 + 0.5*y3 + 1 + np.random.normal(0, 0.2, 15)
        
        dots_3d = VGroup()
        for x, y, z in zip(x3, y3, z3):
            # c2p ya calcula la posición basada en los ejes desplazados
            dots_3d.add(Dot3D(point=axes_3d.c2p(x, y, z), color=YELLOW, radius=0.08))
            
        self.play(FadeIn(dots_3d))
        
        # Plano
        plane = Surface(
            lambda u, v: axes_3d.c2p(u, v, 0.5*u + 0.5*v + 1),
            u_range=[0, 6], v_range=[0, 6], resolution=(8, 8),
            fill_opacity=0.3, stroke_width=0.5, color=BLUE
        )
        self.play(Create(plane))
        
        # Rotación ambiente
        self.begin_ambient_camera_rotation(rate=0.2)
        self.wait(4)
        self.stop_ambient_camera_rotation()
        
        # Limpiar 3D y volver a 2D (Reseteamos zoom a 1)
        self.play(FadeOut(dots_3d), FadeOut(plane), FadeOut(axes_3d), FadeOut(labels_3d), FadeOut(title_3d))
        self.move_camera(phi=0, theta=-90 * DEGREES, zoom=1, run_time=1.5)
        
        # ---------------------------------------------------------
        # ESCENA 4 & 5: OVERFITTING Y REGULARIZACIÓN
        # ---------------------------------------------------------
        # Volver a ejes 2D
        axes = Axes(x_range=[0, 6, 1], y_range=[-2, 12, 2], x_length=8, y_length=6, axis_config={"include_numbers":True})
        self.play(Create(axes))
        
        # Datos para overfitting
        x_reg = np.linspace(0.5, 5.5, 12)
        y_reg = 2 + 0.5 * x_reg**2 - 0.1 * x_reg**3 + np.random.normal(0, 1.5, size=len(x_reg))
        
        dots_reg = VGroup()
        for x, y in zip(x_reg, y_reg): dots_reg.add(Dot(axes.c2p(x, y)))
        self.play(FadeIn(dots_reg))
        
        # 1. Overfitting (Polinomio Grado 10)
        pipe_over = make_pipeline(PolynomialFeatures(10), LinearRegression())
        pipe_over.fit(x_reg.reshape(-1, 1), y_reg)
        def pred_over(x): return pipe_over.predict([[x]]).item()
        
        curve_over = axes.plot(pred_over, color=RED, x_range=[0.5, 5.5])
        label_over = Text("Overfitting", color=RED, font_size=24).next_to(curve_over, UP)
        
        self.play(Create(curve_over), Write(label_over))
        self.wait(3)
        
        # 2. Ridge (L2)
        pipe_ridge = make_pipeline(PolynomialFeatures(10), Ridge(alpha=5))
        pipe_ridge.fit(x_reg.reshape(-1, 1), y_reg)
        def pred_ridge(x): return pipe_ridge.predict([[x]]).item()
        
        curve_ridge = axes.plot(pred_ridge, color=GREEN, x_range=[0.5, 5.5])
        label_ridge = Text("Ridge (L2)", color=GREEN, font_size=24).next_to(curve_ridge, UP)
        
        self.play(Transform(curve_over, curve_ridge), Transform(label_over, label_ridge))
        self.wait(3)
        
        # 3. Lasso (L1)
        pipe_lasso = make_pipeline(PolynomialFeatures(10), Lasso(alpha=0.1, max_iter=10000))
        pipe_lasso.fit(x_reg.reshape(-1, 1), y_reg)
        def pred_lasso(x): return pipe_lasso.predict([[x]]).item()
        
        curve_lasso = axes.plot(pred_lasso, color=ORANGE, x_range=[0.5, 5.5])
        label_lasso = Text("Lasso (L1)", color=ORANGE, font_size=24).next_to(curve_lasso, UP)
        
        self.play(Transform(curve_over, curve_lasso), Transform(label_over, label_lasso))
        self.wait(3)
        
        self.play(FadeOut(curve_over), FadeOut(label_over), FadeOut(dots_reg), FadeOut(axes))

        # ---------------------------------------------------------
        # ESCENA 6: MÉTRICAS (R2 vs R2 Adj)
        # ---------------------------------------------------------
        r2_text = Text("Métricas de Evaluación", font_size=36).to_edge(UP)
        self.play(Write(r2_text))
        
        # Representación de métricas
        r2_eq = MathTex(r"R^2", color=WHITE).shift(LEFT*2 + UP)
        r2_adj_eq = MathTex(r"R^2_{adj}", color=YELLOW).shift(LEFT*2 + DOWN)
        
        val_r2 = DecimalNumber(0.85).next_to(r2_eq, RIGHT)
        val_adj = DecimalNumber(0.83, color=YELLOW).next_to(r2_adj_eq, RIGHT)
        
        note = Text("Añadiendo variables basura...", font_size=24, color=RED).to_edge(RIGHT)
        
        self.play(Write(r2_eq), Write(val_r2), Write(r2_adj_eq), Write(val_adj))
        self.wait(1)
        self.play(FadeIn(note))
        
        # Simular efecto de añadir ruido
        self.play(
            ChangeDecimalToValue(val_r2, 0.86),      # R2 sube (engañoso)
            ChangeDecimalToValue(val_adj, 0.60),     # R2 Adj baja (verdad)
            run_time=3
        )
        self.wait(3)
        
        # ---------------------------------------------------------
        # CONCLUSIÓN
        # ---------------------------------------------------------
        self.play(FadeOut(r2_eq), FadeOut(val_r2), FadeOut(r2_adj_eq), FadeOut(val_adj), FadeOut(note), FadeOut(r2_text))
        
        final = Text("¡Gracias!", font_size=48, color=BLUE)
        self.play(Write(final))
        self.wait(3)