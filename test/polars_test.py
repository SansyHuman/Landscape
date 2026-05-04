from common.balanced_sample_tool import TheorySampler


sampler = TheorySampler('../landscape_all.csv')
print(sampler.a_range())
print(sampler.c_range())
sampler.draw_central_charge_histogram(1)
