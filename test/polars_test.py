from common.balanced_sample_tool import TheorySampler


sampler = TheorySampler('../landscape_all.csv')
print(sampler.a_range())
print(sampler.c_range())
sampler.draw_central_charge_histogram(0.1)
print(sampler.get_theory_stats())

balanced_sample = sampler.get_balanced_sample((0.5, 1.5), (0.5, 1.5), 50)
print(balanced_sample.get_theory_stats())
