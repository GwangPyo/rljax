from conditional_flow import FlowSequential, SigmoidFlowLayer
import haiku as hk
import jax
import jax.numpy as jnp


class RiskProposalNetwork(hk.Module):
    def __init__(self,
                 net_arch=(16, 16, 16, 16),
                 name='risk_proposal_net',
                 ):
        super(RiskProposalNetwork, self).__init__(name=name)
        self.net_arch = net_arch
        self.flow = FlowSequential(
            net_arch=net_arch,
            dim=1,
            outlayer=SigmoidFlowLayer
        )

    def __call__(self, observations, noise):
        observations = jnp.expand_dims(observations, axis=-2,)
        noise = jnp.expand_dims(noise, axis=-1)
        init, (_, forward, inverse, forward_with_params) = self.flow(observations, noise)
        proposal, log_prob, params = forward_with_params(noise, feature=observations)
        proposal = jax.lax.sort(proposal, dimension=-2)
        return proposal.squeeze(-1), log_prob.sum(axis=-2), params.squeeze(-2)


