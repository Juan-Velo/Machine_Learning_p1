from manim import *
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures

class RegressionMasterclass(Scene):
    def construct(self):
        # --- CONFIGURACIÓN ESTÉTICA ---
        self.camera.background_color = "#0F172A" # Slate 900 (Azul oscuro elegante)
        
        # Datos Sintéticos (Semilla fija para consistencia)
        np.random.seed(42)
        self.x_vals = np.linspace(1, 9, 15)
        # Una curva cuadrática con ruido
        self.y_vals = 0.5 * (self.x_vals - 5)**2 + 2 + np.random.normal(0, 0.8, 15)
        
        # --- SECUENCIA ---
        self.intro_scene()
        self.linear_failure()
        self.basis_functions()
        self.overfitting_chaos()
        self.regularization_fix()
        self.outro_metrics()

    def intro_scene(self):
        # 0:00 - 0:15
        title = Text("Regression Analysis", font_size=48, weight=BOLD).to_edge(UP)
        subtitle = Text("From Linear Bias to Regularized Robustness", font_size=32, color=BLUE_C).next_to(title, DOWN)
        
        line = Line(LEFT*4, RIGHT*4, color=BLUE_E).next_to(subtitle, DOWN)
        
        authors = VGroup(
            Text("Ladera La Torre, Fabricio Godofredo", font_size=24, color=GRAY_B),
            Text("Velo Poma, Juan David", font_size=24, color=GRAY_B)
        ).arrange(DOWN, buff=0.2).next_to(line, DOWN, buff=0.5)

        self.play(Write(title), FadeIn(subtitle, shift=UP))
        self.play(Create(line), Write(authors))
        self.wait(6) # Tiempo para leer la intro
        self.play(FadeOut(Group(title, subtitle, line, authors)))

    def linear_failure(self):
        # 0:15 - 0:40
        # Ejes
        axes = Axes(
            x_range=[0, 10, 1], 
            y_range=[-2, 12, 2], 
            axis_config={"color": GREY, "include_numbers": False},
            x_length=7, y_length=5
        ).to_edge(LEFT, buff=1)
        
        labels = axes.get_axis_labels(x_label="Area (x)", y_label="Price (y)")
        self.play(Create(axes), Write(labels))

        # Puntos
        points = VGroup(*[Dot(axes.c2p(x, y), color=WHITE, radius=0.08) for x, y in zip(self.x_vals, self.y_vals)])
        self.play(FadeIn(points, lag_ratio=0.1))
        
        # Modelo Lineal
        model = LinearRegression().fit(self.x_vals.reshape(-1, 1), self.y_vals)
        line = axes.plot(lambda t: model.predict([[t]])[0], color=RED, x_range=[0, 10])
        
        eq_linear = MathTex(r"h_\theta(x) = \theta_0 + \theta_1 x", color=RED).to_corner(UR)
        
        self.play(Create(line), Write(eq_linear))
        self.wait(5) # "Buscamos el vector de pesos..."

        # Guardar para la siguiente escena
        self.axes = axes
        self.points = points
        self.line_linear = line
        self.eq_current = eq_linear

    def basis_functions(self):
        # 0:40 - 1:20
        # Texto Underfitting
        fail_text = Text("Underfitting (High Bias)", font_size=36, color=RED).next_to(self.axes, DOWN)
        self.play(Write(fail_text))
        self.wait(4) # "El mercado tiene curvas..."

        # Transformación de la ecuación
        self.play(FadeOut(fail_text))
        
        eq_poly = MathTex(r"h(x) = \theta_0 + \theta_1 x + \theta_2 x^2", color=GREEN).to_corner(UR)
        basis_map = MathTex(r"\phi(x) \rightarrow [1, x, x^2]", color=GREEN_B, font_size=32).next_to(eq_poly, DOWN)
        
        self.play(TransformMatchingTex(self.eq_current, eq_poly))
        self.play(FadeIn(basis_map))
        
        # Modelo Polinomial (Grado 2)
        poly2 = PolynomialFeatures(degree=2)
        x_poly2 = poly2.fit_transform(self.x_vals.reshape(-1, 1))
        model2 = LinearRegression().fit(x_poly2, self.y_vals)
        
        curve = self.axes.plot(
            lambda t: model2.predict(poly2.transform([[t]]))[0], 
            color=GREEN, x_range=[0, 10]
        )
        
        self.play(ReplacementTransform(self.line_linear, curve))
        self.wait(8) # Explicación de linealidad en parámetros

        self.curve_ok = curve
        self.eq_current = eq_poly
        self.basis_map = basis_map

    def overfitting_chaos(self):
        # 1:20 - 1:55
        # Transición a caos
        warning = Text("Degree = 15", font_size=32, color=ORANGE).next_to(self.eq_current, DOWN, buff=0.5)
        self.play(FadeOut(self.basis_map), Write(warning))
        
        # Modelo Overfitting (Grado alto)
        poly_high = PolynomialFeatures(degree=15)
        x_poly_high = poly_high.fit_transform(self.x_vals.reshape(-1, 1))
        model_high = LinearRegression().fit(x_poly_high, self.y_vals)
        
        curve_bad = self.axes.plot(
            lambda t: model_high.predict(poly_high.transform([[t]]))[0], 
            color=RED, x_range=[1, 9] # Rango acotado para evitar disparos al infinito
        )
        
        overfit_label = Text("Overfitting (High Variance)", font_size=36, color=RED).next_to(self.axes, DOWN)
        
        self.play(ReplacementTransform(self.curve_ok, curve_bad), run_time=2)
        self.play(Write(overfit_label))
        self.wait(6) # "Persigue el ruido..."

        self.curve_bad = curve_bad
        self.overfit_label = overfit_label
        self.x_poly_high = x_poly_high # Guardar datos
        self.warning = warning

    def regularization_fix(self):
        # 1:55 - 2:30
        # Mostrar formula Ridge
        ridge_eq = MathTex(
            r"J(\theta) = MSE + {\lambda} \sum \theta_j^2", 
            tex_to_color_map={"{\lambda}": PURPLE}
        ).to_corner(UR)
        
        self.play(
            FadeOut(self.eq_current), 
            FadeOut(self.warning),
            Write(ridge_eq)
        )
        self.wait(4) # "Agregamos un castigo..."

        # Aplicar Ridge
        model_ridge = Ridge(alpha=10).fit(self.x_poly_high, self.y_vals)
        
        curve_ridge = self.axes.plot(
            lambda t: model_ridge.predict(
                PolynomialFeatures(degree=15).fit_transform([[t]])
            )[0], 
            color=PURPLE, x_range=[0, 10]
        )
        
        ridge_label = Text("L2 Regularization (Ridge)", font_size=36, color=PURPLE).next_to(self.axes, DOWN)
        
        self.play(
            ReplacementTransform(self.curve_bad, curve_ridge),
            ReplacementTransform(self.overfit_label, ridge_label)
        )
        self.wait(6) # "Doma el polinomio..."
        
        # Limpieza final
        self.play(FadeOut(Group(self.axes, self.points, curve_ridge, ridge_label, ridge_eq)))

    def outro_metrics(self):
        # 2:30 - Final
        title = Text("Model Evaluation", font_size=40).to_edge(UP)
        
        # Comparación visual R2
        r2_bad = MathTex(r"R^2 \approx 0.99", color=RED).shift(LEFT*3)
        text_bad = Text("Misleading!", font_size=24, color=RED).next_to(r2_bad, DOWN)
        
        r2_adj = MathTex(r"R^2_{adj} = 1 - (1-R^2)\frac{n-1}{n-p-1}", color=YELLOW).shift(RIGHT*1)
        text_good = Text("Penalizes complexity (p)", font_size=24, color=YELLOW).next_to(r2_adj, DOWN)

        self.play(Write(title))
        self.play(Write(r2_bad), FadeIn(text_bad))
        self.wait(2)
        self.play(Write(r2_adj), FadeIn(text_good))
        self.wait(5)
        
        final_text = Text("Thanks for watching.", font_size=32).move_to(DOWN*2)
        self.play(Write(final_text))
        self.wait(3)