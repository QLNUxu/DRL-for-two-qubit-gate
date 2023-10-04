# DRL-for-two-qubit-gate

In folder \WithoutPriorKnowledge, the most general Hamiltonian for the agent to explore all possibilities
$$H_{DRL} = \frac{1}{2}\sum_{i=1}^2\left(\Omega_i(t)\cos\omega tX+\Omega_i(t)\sin\omega tY+\Delta_i(t)Z\right)+\frac{J(t)}{2}Z_1\otimes Z_2,$$
where the tunable ranges of the parameters are $\Omega\in[0,\Omega_{\max}]$, $\Delta\in[-\Delta_{\max},\Delta_{\max}]$, and $J\in[0,J_{\max}]$.

In folder \GeoMetricKnowledge, the hamiltonian based on the geometric gate knowledge is 
$$H(t) = J \sigma_z \otimes \sigma_z + I \otimes H_c(t) + q(t) \sigma_z \otimes I,$$
where the first part on the right hand side is the Ising interaction, the control Hamiltonian of the second part is $H_c(t) = (\Omega e^{-i\omega t}\sigma_+ + \Omega e^{i\omega t}\sigma_- + D\sigma_z)/2.$
