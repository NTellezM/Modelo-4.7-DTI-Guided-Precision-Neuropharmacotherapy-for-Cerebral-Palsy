# Modelo 4.7 — Formulación matemática

## 1. Ecuación de difusión anisotrópica con liberación controlada

El dominio Ω se descompone en dos subdominios: Ω_gel (hidrogel, tag=1)
y Ω_tejido (tejido cerebral, tags=2,3). La concentración C(x,t) evoluciona
según ecuaciones distintas en cada subdominio:

**En el gel (Ω_gel):** liberación volumétrica controlada por permeabilidad P

    ∂C/∂t = ∇·(D_eff·∇C) − k·C + P·(C_res − C)

**En el tejido (Ω_tejido):** difusión anisotrópica pura

    ∂C/∂t = ∇·(D_eff·∇C) − k·C

donde:
- D_eff(x,t) = ρ(x,t)·D_DTI(x) es el tensor de difusión efectivo (3×3 SPD)
- D_DTI(x) proviene de la interpolación log-euclidiana de imágenes DTI
- ρ(x,t) ∈ [0,1] es el campo de plasticidad estructural
- k es la tasa de eliminación del fármaco
- P es la permeabilidad del gel (controlada por el MPC)
- C_res es la concentración del reservorio interno del gel (constante)

El término P·(C_res − C) modela el gel como un compartimento que libera
fármaco a tasa proporcional a P y al gradiente de concentración respecto
al reservorio. P grande → liberación rápida; P pequeño → retención.

**Nota:** Este modelo difiere de una condición de interfaz tipo Robin
(−D·∂C/∂n = P·(C₁−C₂) sobre Γ). El modelo volumétrico se eligió porque:
(a) no requiere facetas de interfaz conformes (simplifica la geometría),
(b) es más estable numéricamente para geles con extensión espacial, y
(c) captura el mecanismo de liberación de hidrogeles porosos donde el
transporte es un efecto de volumen, no solo de superficie.

### Condiciones de frontera

- Neumann homogéneo en ∂Ω: ∂C/∂n = 0 (flujo cero)
- No hay condición de transmisión explícita: la interfaz gel-tejido es
  conforme (nodos compartidos) y la continuidad de C se impone por el
  espacio P1 continuo.

### Condición inicial

Gaussiana centrada en el hidrogel:
    C(x,0) = C₀·exp(−|x−x_c|²/(2σ²))

## 2. Discretización

### Espacial: elementos finitos P1

Espacio V_h = {v ∈ H¹(Ω) : v|_K ∈ P₁ ∀K ∈ T_h}

### Temporal: θ-esquema (Euler implícito θ=1, Crank-Nicolson θ=0.5)

Forma débil en cada paso temporal (con P):

    ∫_Ω (Cⁿ⁺¹·v)/dt dx + θ·∫_Ω D_eff·∇Cⁿ⁺¹·∇v dx + ∫_Ω k·Cⁿ⁺¹·v dx
    + θ·∫_{Ω_gel} P·Cⁿ⁺¹·v dx
    = ∫_Ω (Cⁿ·v)/dt dx − (1−θ)·∫_Ω D_eff·∇Cⁿ·∇v dx
    + ∫_{Ω_gel} P·C_res·v dx − (1−θ)·∫_{Ω_gel} P·Cⁿ·v dx

Los términos de P solo actúan sobre Ω_gel (dx(1) en FEniCS).
P es un df.Constant mutable: al llamar P.assign(new_P) desde el MPC,
las formas UFL recogen el nuevo valor automáticamente en el siguiente solve.

## 3. Interpolación log-euclidiana de tensores DTI

Para un punto FEM p con 8 vecinos vóxel {T_i} y pesos trilineales {w_i}:

    D_interp(p) = exp(Σᵢ wᵢ·log(Tᵢ))

donde log y exp son logaritmo y exponencial matricial vía descomposición espectral:
    log(T) = V·diag(ln λ₁, ln λ₂, ln λ₃)·V^T
    exp(S) = V·diag(e^λ₁, e^λ₂, e^λ₃)·V^T

Garantiza que D_interp es siempre SPD.

## 4. Plasticidad estructural

El campo ρ(x,t) evoluciona en escala de días según:

    ρⁿ⁺¹ = (ρⁿ + α·H(R_avg − R_th)·Δt_p) / (1 + β·Δt_p)

donde:
- R_avg = (1/|Ω_p|·Δt_p)·∫₀^Δt_p ∫_{Ω_p} R(C) dx dt  (promedio espaciotemporal)
- R(C) = C^n / (C₅₀^n + C^n)  (Hill, ocupación de receptores)
- H(·) = 1/(1 + exp(−10·(·)))  (sigmoide suave)
- α: tasa de refuerzo sináptico
- β: tasa de degradación basal
- R_th: umbral de ocupación para activar refuerzo

## 5. Control predictivo (MPC)

### Modelo linealizado

    q(k) = q₀ + S·Δu(k)

donde:
- q = craving (ocupación promedio de receptores en NAc)
- Δu = secuencia de cambios en permeabilidad P
- S = matriz de Toeplitz construida desde la respuesta al escalón

### Problema de optimización

    min_{Δu} ||q₀·1 + S·Δu − q_ref||² + λ||Δu||²
    s.t.  P_min ≤ P_k + Σ Δu_i ≤ P_max
          |Δu_i| ≤ δP_max

Resuelto con SLSQP (scipy). Se aplica solo Δu₀ (receding horizon).

### Escalas de tiempo

| Proceso | Intervalo | Variable |
|---------|-----------|----------|
| Difusión | dt ~ 0.05-1 s | C(x,t) |
| Control MPC | dt_control ~ 3600 s | P |
| Plasticidad | dt_plastic ~ 86400 s | ρ(x,t) |

## 6. Pipeline computacional

    NIfTI (segmentación) → geom_preprocessing → malla XDMF
    NIfTI (DTI)          → dti_mapper (Rust)   → tensores (N,3,3)
    malla + tensores     → fenics_solver        → C(x,t), ρ(x,t), P(t)
