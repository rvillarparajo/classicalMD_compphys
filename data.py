"""
Data from the simulations that will be analyzed in the main
"""

import numpy as np

rho1 = 0.88
T1 = np.array([1.095, 0.94, 0.591])
Verlet1_p = np.array([3.48, 2.72, -0.18])
p1_108 = np.array([3.554997804, 2.568646543, -2.687535663])
unc1_108 = np.array([0.003568313768, 0.006012207838, 0.005120736921])
p1_256 = np.array([3.701346728, 2.901004857, -3.556053403])
unc1_256 = np.array([0.002490006037, 0.002192723524, 0.002668829668])

rho2 = 0.85
T2 = np.array([2.889, 2.202, 1.214, 1.128, 0.88, 0.782, 0.786, 0.76, 0.719, 0.658, 0.591]) 
Verlet2_p= np.array([4.36, 4.2, 3.06, 2.78, 1.64, 0.98, 0.99, 0.78, 0.36, -0.2, -1.2]) 
p2_108= np.array([4.474687199, 4.528320617, 3.373341077, 3.144508067, 1.903475151, 1.700206169, 1.60172936, 1.327564349, 0.7718627249, -2.067591479, -3.41113397]) 
unc2_108= np.array([0.002267472511, 0.002629779241, 0.003733934754, 0.004169674392, 0.004039405779, 0.004352036953, 0.004046481676, 0.004353737816, 0.004864186896, 0.005117231956, 0.004417019083]) 
p2_256 = np.array([4.354807762, 4.252465042, 3.272829065, 2.928915327, 1.695747345, 1.146549114, 1.323397636, 0.8318251308, 0.7461700852, -2.615593046, -3.95600441])
unc2_256 = np.array([0.001634695832, 0.001801479906, 0.002224970803, 0.002593029557, 0.002006418403, 0.002849389893, 0.003073292985, 0.002989979421, 0.003130803548, 0.00336574096, 0.003712161902])

rho3 = 0.75
T3 = np.array([2.849, 1.304, 1.069, 1.071, 0.881, 0.827]) 
Verlet3_p= np.array([3.1, 1.61, 0.9, 0.89, -0.12, -0.54]) 
p3_108= np.array([3.325714209, 2.214148863, 1.153441865, 1.087018798, 0.2373607174, -0.249458144]) 
unc3_108= np.array([0.003034493732, 0.004235780992, 0.003281929825, 0.003449498164, 0.003744564902, 0.004630184095]) 
p3_256 = np.array([3.037068458, 1.748545884, 0.9760893982, 0.8557189563, -0.06613499707, -0.3889606883])
unc3_256 = np.array([0.00130368988, 0.002074942526, 0.002048315474, 0.002654522209, 0.002707944531, 0.003297179403])

rho4 = 0.65
T4 = np.array([2.557, 1.585, 1.036, 0.9]) 
Verlet4_p= np.array([2.14, 1.25, -0.11, -0.74]) 
p4_108= np.array([2.230818482, 1.591364864, 0.1747459488, -0.4983456798]) 
unc4_108= np.array([0.002839371918, 0.003331127788, 0.004838019244, 0.00544501392]) 
p4_256 = np.array([2.258481966, 1.328593353, -0.04510527178, -0.7311715168])
unc4_256 = np.array([0.001502774924, 0.002022510994, 0.003036227641, 0.003059481645])

rho5 = 0.45
T5 = np.array([4.625, 2.935, 1.744, 1.764, 1.71, 1.552]) 
Verlet5_p = np.array([1.68, 1.38, 0.74, 0.76, 0.74, 0.75]) 
p5_108 = np.array([1.687343773, 1.404658924, 0.8596368808, 0.8920130789, 0.9236546735, 0.6974882214]) 
unc5_108 = np.array([0.001561024995, 0.002757039058, 0.002722713503, 0.002741884589, 0.003408629482, 0.003681589252]) 
p5_256 = np.array([1.680767477, 1.39188022, 0.8161587547, 0.8005997172, 0.7613077225, 0.5767445921])
unc5_256 = np.array([0.001172509823, 0.001384266592, 0.001856849406, 0.001740666111, 0.002034184867, 0.001987307722])


cv1_108 = np.array([1.707776639, 2.229752828, 1.741453665])
s1_108 = np.array([0.0041953755, 0.02566052214, 0.005902898727])
cv1_256 = np.array([1.719883583, 1.697602751, 1.778992496])
s1_256 = np.array([0.004613901566, 0.004735204103, 0.006646808371])

TVerlet_cv2 = np.array([2.889, 2.202, 1.214, 1.128, 0.88])
Verlet_cv2 = np.array([0.73, 0.79, 0.95, 0.99, 1.11])
cv2_108 = np.array([1.585907407, 1.604058219, 1.710594518, 1.73791642, 1.685228276, 1.681809091, 1.756469314, 1.697282326, 1.75387229, 1.782632935, 1.752963222])
s2_108 = np.array([0.001677435863, 0.002152898855, 0.005084303728, 0.004432957423, 0.004124975515, 0.004074695614, 0.006011009565, 0.004773126971, 0.006442624014, 0.006967597508, 0.005524497815])
cv2_256 = np.array([1.626048114, 1.630570754, 1.677761541, 1.705048512, 1.668619162, 1.716693111, 1.709999112, 1.74745828, 1.74549618, 1.816202881, 1.82946314])
s2_256 = np.array([0.003083550763, 0.002955786101, 0.004300711871, 0.005118365909, 0.004136700503, 0.005089721801, 0.005239503924, 0.006130278392, 0.005503774534, 0.006823214515, 0.008489542189])

TVerlet_cv3 = np.array([2.849, 0.827])
Verlet_cv3 = np.array([0.56, 0.88])
cv3_108 = np.array([1.588961552, 1.697939173, 1.618002894, 1.624010353, 1.634922625, 1.692766775])
s3_108 = np.array([0.001827593089, 0.005134576359, 0.0028104627, 0.003427273771, 0.003001694803, 0.004350162492])
cv3_256 = np.array([1.577313931, 1.599367708, 1.642558654, 1.69946426, 1.671981489, 1.741833648])
s3_256 = np.array([0.001895295018, 0.002192558162, 0.003910405222, 0.005117547871, 0.003659922927, 0.004926447297])

TVerlet_cv4 = np.array([4.625, 2.935, 1.71, 1.51])
Verlet_cv4 = np.array([0.2, 0.26, 0.28, 0.28])
cv4_108 = np.array([1.5940229, 1.618977642, 1.643374394, 1.682539625])
s4_108 = np.array([0.002056483268, 0.002670935602, 0.002516588266, 0.003821491952])
cv4_256 = np.array([1.565595444, 1.588211953, 1.670969796, 1.735327381])
s4_256 = np.array([0.001321878799, 0.001674240607, 0.003880788844, 0.005460353434])

TVerlet_cv5 = np.array([1.462])
Verlet_cv5 = np.array([0.54])
cv5_108 = np.array([1.535321036, 1.55707765, 1.591823986, 1.597014162, 1.602582625, 1.652642481])
s5_108 = np.array([0.000880479815, 0.001094276699, 0.001807713564, 0.002464419247, 0.002176102872, 0.003876777242])
cv5_256 = np.array([1.53242634, 1.537728006, 1.622712949, 1.593651215, 1.615383987, 1.59552745])
s5_256 = np.array([0.0006663074826, 0.0006826304669, 0.002617470365, 0.001873062062, 0.002489616857, 0.001827369478])



beta_list = [1.0/T1, 1.0/T2, 1.0/T3, 1.0/T4, 1.0/T5]
Verlet_list = [Verlet1_p, Verlet2_p, Verlet3_p, Verlet4_p, Verlet5_p]

p_list_108 = [p1_108, p2_108, p3_108, p4_108, p5_108]
unc_list_108 = [unc1_108, unc2_108, unc3_108, unc4_108, unc5_108]

p_list_256 = [p1_256, p2_256, p3_256, p4_256, p5_256]
unc_list_256 = [unc1_256, unc2_256, unc3_256, unc4_256, unc5_256]

labels = [r'$\rho={}$'.format(rho1), r'$\rho={}$'.format(rho2), r'$\rho={}$'.format(rho3), r'$\rho={}$'.format(rho4), r'$\rho={}$'.format(rho5)]
color_list = ['#332288', '#228833', '#ee7733', '#997700', '#0077bb']
rot_list_108 = [-57 , 2, -30, -32, -29]
pad_list_108 = [-0.1, 0.25, 0, -0.1, 0]
rot_list_256 = [-50 , -11, -30, -37, -29]
pad_list_256 = [-0.1, 0.15, 0, -0.2, 0.05]



beta_list = [1.0/T1, 1.0/T2, 1.0/T3, 1.0/T4, 1.0/T5]
TVerlet_list_cv = [1, TVerlet_cv2, 3, TVerlet_cv4, TVerlet_cv5]
Verlet_list_cv = [1, Verlet_cv2, 3, Verlet_cv4, Verlet_cv5]

cv_list_108 = [cv1_108, cv2_108, cv3_108, cv4_108, cv5_108]
s_list_108 = [s1_108, s2_108, s3_108, s4_108, s5_108]

cv_list_256 = [cv1_256, cv2_256, cv3_256, cv4_256, cv5_256]
s_list_256 = [s1_256, s2_256, s3_256, s4_256, s5_256]


