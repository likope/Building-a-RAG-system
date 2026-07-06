from assistant import Assistant
import re

class Eval:
    def __init__(self):
        self.assistant = Assistant()
    
    def normalize_text(self, text: str):
        """
        f che si occupa di formattare il testo in modo da renderlo piu leggibile.
        """
        text = text.replace("-\n", "")
        return " ".join(text.split())
    

    def evaluate(self, current_state):
        """
        f che valuta la risposta del modello in base a quanto corrisponde al contesto.
        """
        simboli_math = {"∫", "Γ", "√"}
        context = current_state["context"]
        output = current_state["output"]
        cit = re.findall(r'"([^"]*)"', output)
        cit_norm = [self.normalize_text(c) for c in cit]
        context_norm = self.normalize_text(context)
        for c in cit_norm:
            if any(sym in c for sym in simboli_math):
                continue
            elif c not in context_norm:
                return False
        return True
    
    def parser(self, current_state):
        """
        f che valuta la risposta del judge in base a quanto corrisponde alle citazioni presenti nel contesto.
        """
        judge_output = current_state["judge_output"]
        match = re.search(r"\$(\d+),(\d+),(\d+)", judge_output)
        if match:
            accuratezza = int(match.group(1))
            risposta = int(match.group(2))
            riferimenti = int(match.group(3))
            return accuratezza, risposta, riferimenti
        else:
            return None, None, None
        
if __name__ == "__main__":
    eval = Eval()
    stato_alpha = {'input': 'what is "alpha" in rough heston model?', 'history': 'input = what is "alpha" in rough heston model?, output = In the rough Heston model, "alpha" is a parameter, as indicated by "alpha = x (6) ;" and an example value "α = 0.62". It appears in the integral terms defining `Vt`, specifically within the kernel as `(t −s)α−1` and in the denominator with the Gamma function, `Γ(α)`. The full equation for `Vt` is given as "Vt = V0 + 1/Γ(α) ∫_0^t (t −s)α−1γ(θ −Vs) ds + 1/Γ(α) ∫_0^t (t −s)α−1γν √Vs dBs"., judge_output = ', 'output': 'In the rough Heston model, "alpha" is a parameter, as indicated by "alpha = x (6) ;" and an example value "α = 0.62". It appears in the integral terms defining `Vt`, specifically within the kernel as `(t −s)α−1` and in the denominator with the Gamma function, `Γ(α)`. The full equation for `Vt` is given as "Vt = V0 + 1/Γ(α) ∫_0^t (t −s)α−1γ(θ −Vs) ds + 1/Γ(α) ∫_0^t (t −s)α−1γν √Vs dBs".', 'context': '21\n22 % rough\nHeston model\nparameters\n23 nu = x (1) ;\n24 lambda= x (2) ;\n25 rho = x (3) ;\n26 theta = x (4) ;\n27 V0 = x (5) ;\n28 alpha = x (6) ;\n29\n30 % Fractional\nRiccati\nequation\n31 u=1i ∗u ;\n32 F = @(v) (0.5∗( u .∗u−u) + (u∗rho∗nu−lambda ) .∗v + (0.5∗( nu∗v) .^2) ) ;\n33\n34 % M u l t i f a c t o r\nmodel\nc o e f f i c i e n t s\n35 i =[1:n ] ;\n36 c=((r_n^(1−alpha ) −1)∗(r_n^(( alpha −1)∗(1+n/2) ) ) ∗. . .\n37\n(r_n.^((1 −alpha ) .∗i ) ) ) /(gamma( alpha ) ∗gamma(2−alpha ) ) ;\n75, 3.6\nCharacteristic function of Rough Heston\nmodel\nWe set\nV0 = ξθ , ρ =\n1 −β\np\n2(1 + β2)\n, ν =\ns\nθ(1 + β2)\nλµ(1 + β2) , λ = γ\nwhere λ and θ are the same as those in the dynamic of P T. Remark that\nthis implies that ρ ∈(−1/\n√\n2, 1/\n√\n2]. We also write Pt = log(St/S0).\nWe consider the rough Heston model:\ndSt = St\np\nVtdWt\nVt = V0 +\n1\nΓ(α)\nZ t\n0\n(t −s)α−1γ(θ −Vs) ds +\n1\nΓ(α)\nZ t\n0\n(t −s)α−1γν\np\nVs dBs\nThe parameters γ, θ, V0 and ν are positive and play the same role as in the, Appendix A.2.\n5.5\nNumerical illustration\nConsidering the following parameters:\nν = 0.05, γ = 2, ρ = −0.5, θ = 0.04, V0 = 0.04, α = 0.62\nwe compare in ﬁgure 5.1 the ATM skew produced by the classical Heston\nmodel and its rough counterpart. We highligt that the rough Heston model\nis able to reproduce the exploding structure of the skew when the time to\nmaturity goes to zero.\n54', 'judge_output': ''}
    print(eval.evaluate(stato_alpha))
    import copy
    stato_falso = copy.deepcopy(stato_alpha)
    stato_falso["output"] = stato_falso["output"].replace("α = 0.62", "α = 0.99")
    print(eval.evaluate(stato_falso))