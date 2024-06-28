import vkdispatch as vd 
from matplotlib import pyplot as plt
import numpy as np
import sys
import tf_calc
import tqdm
import time

#vd.initialize(log_level=vd.LogLevel.INFO)
vd.make_context(devices=[0], queue_families=[[0]])

import test_potential as tp
import test_scope as ts
import test_correlator as tc
import test_utils as tu

#phi_values = np.arange(0, 360, 2.5)
#theta_values = np.arange(0, 180, 2.5)
#psi_values = np.arange(0, 360, 1.5)

#phi_values = np.arange(100, 200, 2.5)
#theta_values = np.arange(70, 100, 2.5)
#psi_values = np.arange(280, 340, 1.5)

phi_values = np.arange(150, 200, 2.5)
theta_values = np.arange(70, 90, 2.5)
psi_values = np.arange(320, 340, 1.5)

defocus_values = np.arange(10000, 16000, 400)
test_values = (np.array(np.meshgrid(phi_values, theta_values, psi_values, defocus_values)).T.reshape(-1, 4))

template_size = (512, 512)# (380, 380)

sss = time.time()

potential_generator = tp.TemplatePotential(tu.load_coords(sys.argv[3]), template_size, 200, 0.3)
scope = ts.Scope(template_size, tf_calc.prepareTF(template_size, 1.056, 0), tf_calc.get_sigmaE(300e3), -0.07)
correlator = tc.Correlator(template_size, tu.load_image(sys.argv[2]))

print("Init time:", time.time() - sss)

work_buffer = vd.Buffer(template_size, vd.complex64)

sss = time.time()

cmd_list = vd.CommandList()

potential_generator.record(cmd_list, work_buffer)
scope.record(cmd_list, work_buffer)
correlator.record(cmd_list, work_buffer)

print("Record time:", time.time() - sss)

def set_params(params):
    potential_generator.set_rotation_matrix(tu.get_rotation_matrix(params[1][:3], [0, 0]))
    scope.set_defocus(params[1][3])
    correlator.set_index(params[0])

batch_size = 100
status_bar = tqdm.tqdm(total=test_values.shape[0])
for data in cmd_list.iter_batched_params(set_params, enumerate(test_values), batch_size=batch_size):
    cmd_list.submit_any(data)
    status_bar.update(batch_size)
status_bar.close()

final_max_cross, best_index_result, index_of_max, final_index = tu.aggregate_results(correlator)

print("Found max at:", index_of_max)
print("Max cross correlation:", final_max_cross[index_of_max])
print("Final index:", final_index)
print("Phi:", test_values[final_index][0])
print("Theta:", test_values[final_index][1])
print("Psi:", test_values[final_index][2])
print("Defocus:", test_values[final_index][3])

file_out = sys.argv[1]

potential_generator.record(None, work_buffer, rot_matrix=tu.get_rotation_matrix([test_values[final_index][0], test_values[final_index][1], test_values[final_index][2]]))
np.save(file_out + "_match.npy", work_buffer.read()[0])
scope.record(None, work_buffer, defocus=test_values[final_index][3])
np.save(file_out + "_match_defocused.npy", work_buffer.read()[0])

params_result = test_values[best_index_result]

np.save(file_out + "_mip.npy", final_max_cross)
np.save(file_out + "_best_index.npy", best_index_result)
np.save(file_out + "_phi.npy", params_result[:, :, 0])
np.save(file_out + "_theta.npy", params_result[:, :, 1])
np.save(file_out + "_psi.npy", params_result[:, :, 2])
np.save(file_out + "_defocus.npy", params_result[:, :, 3])

plt.imshow(final_max_cross)
plt.colorbar()
plt.show()

# Do 3D plot of MIP
#from matplotlib import cm
#from matplotlib.ticker import LinearLocator
#fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#X, Y = np.meshgrid(range(final_max_cross.shape[0]), range(final_max_cross.shape[1]))
#surf = ax.plot_surface(X, Y, final_max_cross, cmap=cm.coolwarm,
#                       linewidth=0, antialiased=False)
#ax.zaxis.set_major_locator(LinearLocator(10))
#ax.zaxis.set_major_formatter('{x:.02f}')
#fig.colorbar(surf, shrink=0.5, aspect=5)
#plt.show()
