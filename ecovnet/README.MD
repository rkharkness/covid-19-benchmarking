# ECOVNet
The perilous COVID-19 disease puts the world in an exotic state of emergency since it overwhelms
the global healthcare system. Consequently, people’s lives are at greater risk due to high mortality
as there is no effective vaccine as a precaution against contracting the occurrence of COVID-19
disease. Also, researchers all over the world are working to develop corresponding vaccines, and
at the same time striving to obtain something effective but pragmatic screening technologies, such
as medical imaging. To combat this disease, one of the preeminent screening techniques may be
chest X-ray, as it has many historical credentials related to lung diseases that can provide clinical
insights. This paper proposed an ensemble of deep convolutional neural networks (CNN) based on
EfficientNet, named ECOVNet, to detect COVID-19 using a large chest X-ray data set. At first, the
open-access large chest X-ray collection is augmented, and then ImageNet pre-trained weights for
EfficientNet is transferred with some customized fine-tuning top layers that are trained, followed
by an ensemble of model snapshots to classify chest X-rays corresponding to COVID-19, normal,
and pneumonia. The predictions of the model snapshots, which are created during a single training,
are combined through two ensemble strategies, i.e., hard ensemble and soft ensemble to ameliorate
classification performance and generalization in the related task of classifying chest X-rays. In
addition, a visualization technique is incorporated in the proposed method to highlight areas that
distinguish categories, thereby enhancing the understanding of primal components related to COVID19 infection. Empirical evaluations show that the ensemble strategy (especially soft ensemble) can
significantly improve prediction performance with 97% accuracy, while the precision and recall rate
of detecting COVID-19 are both 100%. We believe that ECOVNet can strengthen the resistance
to COVID-19 disease, and more broadly, it will propel towards a fully automated and efficacious
COVID-19 detection system.

