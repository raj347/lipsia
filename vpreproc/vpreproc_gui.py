# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Preprocessing.ui'
#
# Created: Thu Mar  8 15:57:41 2012
#      by: PyQt4 UI code generator 4.7.3
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

class Ui_Fenster(object):
    def setupUi(self, Fenster):
        Fenster.setObjectName("Fenster")
        Fenster.resize(672, 746)
        Fenster.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.tabWidget = QtGui.QTabWidget(Fenster)
        self.tabWidget.setEnabled(True)
        self.tabWidget.setGeometry(QtCore.QRect(10, 10, 651, 581))
        self.tabWidget.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.tabWidget.setObjectName("tabWidget")
        self.tab_general = QtGui.QWidget()
        self.tab_general.setObjectName("tab_general")
        self.dsb_repetition_time = QtGui.QDoubleSpinBox(self.tab_general)
        self.dsb_repetition_time.setGeometry(QtCore.QRect(180, 290, 62, 27))
        self.dsb_repetition_time.setMaximum(600.0)
        self.dsb_repetition_time.setObjectName("dsb_repetition_time")
        self.l_cutoff_lp_s = QtGui.QLabel(self.tab_general)
        self.l_cutoff_lp_s.setGeometry(QtCore.QRect(570, 310, 16, 31))
        self.l_cutoff_lp_s.setObjectName("l_cutoff_lp_s")
        self.cb_spatial_filtering = QtGui.QCheckBox(self.tab_general)
        self.cb_spatial_filtering.setGeometry(QtCore.QRect(310, 140, 191, 22))
        self.cb_spatial_filtering.setObjectName("cb_spatial_filtering")
        self.b_add_image_directory = QtGui.QPushButton(self.tab_general)
        self.b_add_image_directory.setGeometry(QtCore.QRect(100, 230, 101, 27))
        self.b_add_image_directory.setAutoDefault(False)
        self.b_add_image_directory.setObjectName("b_add_image_directory")
        self.cb_debug_output = QtGui.QCheckBox(self.tab_general)
        self.cb_debug_output.setGeometry(QtCore.QRect(460, 370, 151, 22))
        self.cb_debug_output.setObjectName("cb_debug_output")
        self.cb_slicetime_correction = QtGui.QCheckBox(self.tab_general)
        self.cb_slicetime_correction.setGeometry(QtCore.QRect(310, 60, 151, 22))
        self.cb_slicetime_correction.setObjectName("cb_slicetime_correction")
        self.l_preprocessing_steps = QtGui.QLabel(self.tab_general)
        self.l_preprocessing_steps.setGeometry(QtCore.QRect(300, 20, 151, 21))
        self.l_preprocessing_steps.setObjectName("l_preprocessing_steps")
        self.l_cutoff_high_pass = QtGui.QLabel(self.tab_general)
        self.l_cutoff_high_pass.setGeometry(QtCore.QRect(310, 310, 51, 31))
        self.l_cutoff_high_pass.setObjectName("l_cutoff_high_pass")
        self.sb_lp_cutoff = QtGui.QSpinBox(self.tab_general)
        self.sb_lp_cutoff.setEnabled(False)
        self.sb_lp_cutoff.setGeometry(QtCore.QRect(510, 310, 55, 27))
        self.sb_lp_cutoff.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.sb_lp_cutoff.setMaximum(600)
        self.sb_lp_cutoff.setProperty("value", 3)
        self.sb_lp_cutoff.setObjectName("sb_lp_cutoff")
        self.cb_set_repetition = QtGui.QCheckBox(self.tab_general)
        self.cb_set_repetition.setEnabled(True)
        self.cb_set_repetition.setGeometry(QtCore.QRect(20, 290, 151, 22))
        self.cb_set_repetition.setAutoFillBackground(False)
        self.cb_set_repetition.setChecked(True)
        self.cb_set_repetition.setTristate(False)
        self.cb_set_repetition.setObjectName("cb_set_repetition")
        self.l_input_files = QtGui.QLabel(self.tab_general)
        self.l_input_files.setGeometry(QtCore.QRect(10, 10, 71, 17))
        self.l_input_files.setObjectName("l_input_files")
        self.l_mask_options = QtGui.QLabel(self.tab_general)
        self.l_mask_options.setGeometry(QtCore.QRect(300, 420, 111, 21))
        self.l_mask_options.setObjectName("l_mask_options")
        self.cb_low_pass = QtGui.QCheckBox(self.tab_general)
        self.cb_low_pass.setGeometry(QtCore.QRect(460, 290, 151, 22))
        self.cb_low_pass.setObjectName("cb_low_pass")
        self.cb_fieldmap_correction = QtGui.QCheckBox(self.tab_general)
        self.cb_fieldmap_correction.setGeometry(QtCore.QRect(310, 40, 151, 22))
        self.cb_fieldmap_correction.setObjectName("cb_fieldmap_correction")
        self.cb_write_logfile = QtGui.QCheckBox(self.tab_general)
        self.cb_write_logfile.setGeometry(QtCore.QRect(460, 390, 151, 22))
        self.cb_write_logfile.setObjectName("cb_write_logfile")
        self.l_cutoff_low_pass = QtGui.QLabel(self.tab_general)
        self.l_cutoff_low_pass.setGeometry(QtCore.QRect(460, 310, 51, 31))
        self.l_cutoff_low_pass.setObjectName("l_cutoff_low_pass")
        self.l_miscellaneous = QtGui.QLabel(self.tab_general)
        self.l_miscellaneous.setGeometry(QtCore.QRect(460, 350, 111, 16))
        self.l_miscellaneous.setObjectName("l_miscellaneous")
        self.lw_input_files = QtGui.QListWidget(self.tab_general)
        self.lw_input_files.setGeometry(QtCore.QRect(10, 30, 281, 192))
        self.lw_input_files.setObjectName("lw_input_files")
        self.l_spatial_filtering = QtGui.QLabel(self.tab_general)
        self.l_spatial_filtering.setGeometry(QtCore.QRect(300, 350, 111, 20))
        self.l_spatial_filtering.setObjectName("l_spatial_filtering")
        self.l_spatial_filtering_mm = QtGui.QLabel(self.tab_general)
        self.l_spatial_filtering_mm.setGeometry(QtCore.QRect(420, 370, 31, 31))
        self.l_spatial_filtering_mm.setObjectName("l_spatial_filtering_mm")
        self.cb_atlas_registration = QtGui.QCheckBox(self.tab_general)
        self.cb_atlas_registration.setGeometry(QtCore.QRect(310, 100, 161, 22))
        self.cb_atlas_registration.setObjectName("cb_atlas_registration")
        self.l_output = QtGui.QLabel(self.tab_general)
        self.l_output.setGeometry(QtCore.QRect(10, 350, 51, 16))
        self.l_output.setObjectName("l_output")
        self.cb_show_mask = QtGui.QCheckBox(self.tab_general)
        self.cb_show_mask.setEnabled(False)
        self.cb_show_mask.setGeometry(QtCore.QRect(330, 200, 201, 22))
        self.cb_show_mask.setObjectName("cb_show_mask")
        self.sb_hp_cutoff = QtGui.QSpinBox(self.tab_general)
        self.sb_hp_cutoff.setGeometry(QtCore.QRect(360, 310, 55, 27))
        self.sb_hp_cutoff.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.sb_hp_cutoff.setMaximum(600)
        self.sb_hp_cutoff.setProperty("value", 90)
        self.sb_hp_cutoff.setObjectName("sb_hp_cutoff")
        self.l_fwhm = QtGui.QLabel(self.tab_general)
        self.l_fwhm.setGeometry(QtCore.QRect(310, 370, 51, 31))
        self.l_fwhm.setObjectName("l_fwhm")
        self.cb_show_registration_results = QtGui.QCheckBox(self.tab_general)
        self.cb_show_registration_results.setEnabled(False)
        self.cb_show_registration_results.setGeometry(QtCore.QRect(330, 120, 281, 22))
        self.cb_show_registration_results.setChecked(False)
        self.cb_show_registration_results.setObjectName("cb_show_registration_results")
        self.cb_create_mask = QtGui.QCheckBox(self.tab_general)
        self.cb_create_mask.setGeometry(QtCore.QRect(310, 180, 201, 22))
        self.cb_create_mask.setObjectName("cb_create_mask")
        self.b_add_image_file = QtGui.QPushButton(self.tab_general)
        self.b_add_image_file.setGeometry(QtCore.QRect(10, 230, 81, 27))
        self.b_add_image_file.setCheckable(False)
        self.b_add_image_file.setAutoRepeatDelay(300)
        self.b_add_image_file.setObjectName("b_add_image_file")
        self.b_remove_file = QtGui.QPushButton(self.tab_general)
        self.b_remove_file.setGeometry(QtCore.QRect(210, 230, 81, 27))
        self.b_remove_file.setObjectName("b_remove_file")
        self.l_cutoff_hp_s = QtGui.QLabel(self.tab_general)
        self.l_cutoff_hp_s.setGeometry(QtCore.QRect(420, 310, 16, 31))
        self.l_cutoff_hp_s.setObjectName("l_cutoff_hp_s")
        self.cb_movement_correction = QtGui.QCheckBox(self.tab_general)
        self.cb_movement_correction.setGeometry(QtCore.QRect(310, 80, 161, 22))
        self.cb_movement_correction.setObjectName("cb_movement_correction")
        self.l_directory = QtGui.QLabel(self.tab_general)
        self.l_directory.setGeometry(QtCore.QRect(20, 370, 71, 17))
        self.l_directory.setObjectName("l_directory")
        self.l_temporal_filtering = QtGui.QLabel(self.tab_general)
        self.l_temporal_filtering.setGeometry(QtCore.QRect(300, 270, 151, 21))
        self.l_temporal_filtering.setObjectName("l_temporal_filtering")
        self.cb_high_pass = QtGui.QCheckBox(self.tab_general)
        self.cb_high_pass.setEnabled(True)
        self.cb_high_pass.setGeometry(QtCore.QRect(310, 290, 151, 22))
        self.cb_high_pass.setAutoFillBackground(False)
        self.cb_high_pass.setChecked(True)
        self.cb_high_pass.setTristate(False)
        self.cb_high_pass.setObjectName("cb_high_pass")
        self.sb_fwhm = QtGui.QSpinBox(self.tab_general)
        self.sb_fwhm.setGeometry(QtCore.QRect(360, 370, 55, 27))
        self.sb_fwhm.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.sb_fwhm.setMaximum(600)
        self.sb_fwhm.setProperty("value", 6)
        self.sb_fwhm.setObjectName("sb_fwhm")
        self.cb_temporal_filtering = QtGui.QCheckBox(self.tab_general)
        self.cb_temporal_filtering.setGeometry(QtCore.QRect(310, 160, 231, 22))
        self.cb_temporal_filtering.setObjectName("cb_temporal_filtering")
        self.le_directory = QtGui.QLineEdit(self.tab_general)
        self.le_directory.setGeometry(QtCore.QRect(20, 390, 231, 27))
        self.le_directory.setObjectName("le_directory")
        self.le_prefix = QtGui.QLineEdit(self.tab_general)
        self.le_prefix.setGeometry(QtCore.QRect(20, 450, 231, 27))
        self.le_prefix.setObjectName("le_prefix")
        self.l_prefix = QtGui.QLabel(self.tab_general)
        self.l_prefix.setGeometry(QtCore.QRect(20, 430, 59, 17))
        self.l_prefix.setObjectName("l_prefix")
        self.rb_create_average_mask = QtGui.QRadioButton(self.tab_general)
        self.rb_create_average_mask.setGeometry(QtCore.QRect(310, 450, 191, 22))
        self.rb_create_average_mask.setChecked(True)
        self.rb_create_average_mask.setObjectName("rb_create_average_mask")
        self.rb_create_mask_subject = QtGui.QRadioButton(self.tab_general)
        self.rb_create_mask_subject.setGeometry(QtCore.QRect(310, 470, 231, 22))
        self.rb_create_mask_subject.setObjectName("rb_create_mask_subject")
        self.l_maximum_number_of_voxel = QtGui.QLabel(self.tab_general)
        self.l_maximum_number_of_voxel.setGeometry(QtCore.QRect(310, 500, 181, 31))
        self.l_maximum_number_of_voxel.setObjectName("l_maximum_number_of_voxel")
        self.sb_max_num_voxel = QtGui.QSpinBox(self.tab_general)
        self.sb_max_num_voxel.setGeometry(QtCore.QRect(490, 500, 151, 27))
        self.sb_max_num_voxel.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.sb_max_num_voxel.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.sb_max_num_voxel.setMaximum(1000000000)
        self.sb_max_num_voxel.setProperty("value", 50000)
        self.sb_max_num_voxel.setObjectName("sb_max_num_voxel")
        self.b_output_directory = QtGui.QPushButton(self.tab_general)
        self.b_output_directory.setGeometry(QtCore.QRect(250, 390, 26, 25))
        self.b_output_directory.setObjectName("b_output_directory")
        self.tabWidget.addTab(self.tab_general, "")
        self.tab_reg_settings = QtGui.QWidget()
        self.tab_reg_settings.setAccessibleName("")
        self.tab_reg_settings.setObjectName("tab_reg_settings")
        self.l_atlas_registration = QtGui.QLabel(self.tab_reg_settings)
        self.l_atlas_registration.setGeometry(QtCore.QRect(10, 10, 151, 21))
        self.l_atlas_registration.setObjectName("l_atlas_registration")
        self.l_template = QtGui.QLabel(self.tab_reg_settings)
        self.l_template.setGeometry(QtCore.QRect(20, 30, 71, 17))
        self.l_template.setObjectName("l_template")
        self.comboBox_template = QtGui.QComboBox(self.tab_reg_settings)
        self.comboBox_template.setGeometry(QtCore.QRect(20, 50, 191, 27))
        self.comboBox_template.setObjectName("comboBox_template")
        self.b_view = QtGui.QPushButton(self.tab_reg_settings)
        self.b_view.setGeometry(QtCore.QRect(220, 50, 61, 27))
        self.b_view.setObjectName("b_view")
        self.l_functional_anatomical_image = QtGui.QLabel(self.tab_reg_settings)
        self.l_functional_anatomical_image.setGeometry(QtCore.QRect(20, 90, 251, 17))
        self.l_functional_anatomical_image.setObjectName("l_functional_anatomical_image")
        self.lw_functional_anatomical_image = QtGui.QListWidget(self.tab_reg_settings)
        self.lw_functional_anatomical_image.setGeometry(QtCore.QRect(20, 110, 601, 161))
        self.lw_functional_anatomical_image.setObjectName("lw_functional_anatomical_image")
        self.b_remove_anatomical_image = QtGui.QPushButton(self.tab_reg_settings)
        self.b_remove_anatomical_image.setGeometry(QtCore.QRect(440, 280, 181, 27))
        self.b_remove_anatomical_image.setObjectName("b_remove_anatomical_image")
        self.b_add_anatomical_image = QtGui.QPushButton(self.tab_reg_settings)
        self.b_add_anatomical_image.setGeometry(QtCore.QRect(20, 280, 181, 27))
        self.b_add_anatomical_image.setObjectName("b_add_anatomical_image")
        self.l_registration_pipeline = QtGui.QLabel(self.tab_reg_settings)
        self.l_registration_pipeline.setGeometry(QtCore.QRect(20, 320, 141, 17))
        self.l_registration_pipeline.setObjectName("l_registration_pipeline")
        self.comboBox_interpol_func = QtGui.QComboBox(self.tab_reg_settings)
        self.comboBox_interpol_func.setGeometry(QtCore.QRect(160, 490, 191, 21))
        self.comboBox_interpol_func.setObjectName("comboBox_interpol_func")
        self.l_interpolator_function = QtGui.QLabel(self.tab_reg_settings)
        self.l_interpolator_function.setGeometry(QtCore.QRect(20, 490, 141, 21))
        self.l_interpolator_function.setObjectName("l_interpolator_function")
        self.l_output_resolution = QtGui.QLabel(self.tab_reg_settings)
        self.l_output_resolution.setGeometry(QtCore.QRect(410, 490, 121, 21))
        self.l_output_resolution.setObjectName("l_output_resolution")
        self.le_output_resolution = QtGui.QLineEdit(self.tab_reg_settings)
        self.le_output_resolution.setGeometry(QtCore.QRect(550, 490, 71, 21))
        self.le_output_resolution.setAlignment(QtCore.Qt.AlignCenter)
        self.le_output_resolution.setObjectName("le_output_resolution")
        self.fr_reg_pipe_1 = QtGui.QFrame(self.tab_reg_settings)
        self.fr_reg_pipe_1.setGeometry(QtCore.QRect(20, 340, 161, 131))
        self.fr_reg_pipe_1.setFrameShape(QtGui.QFrame.StyledPanel)
        self.fr_reg_pipe_1.setFrameShadow(QtGui.QFrame.Raised)
        self.fr_reg_pipe_1.setObjectName("fr_reg_pipe_1")
        self.cb_rigid_registration = QtGui.QCheckBox(self.fr_reg_pipe_1)
        self.cb_rigid_registration.setGeometry(QtCore.QRect(10, 10, 141, 22))
        self.cb_rigid_registration.setObjectName("cb_rigid_registration")
        self.l_max_iterations_p1 = QtGui.QLabel(self.fr_reg_pipe_1)
        self.l_max_iterations_p1.setGeometry(QtCore.QRect(10, 30, 141, 21))
        self.l_max_iterations_p1.setObjectName("l_max_iterations_p1")
        self.sb_max_iterations_p1 = QtGui.QSpinBox(self.fr_reg_pipe_1)
        self.sb_max_iterations_p1.setEnabled(False)
        self.sb_max_iterations_p1.setGeometry(QtCore.QRect(10, 50, 141, 21))
        self.sb_max_iterations_p1.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.sb_max_iterations_p1.setMaximum(1000)
        self.sb_max_iterations_p1.setProperty("value", 500)
        self.sb_max_iterations_p1.setObjectName("sb_max_iterations_p1")
        self.cb_prealign_images = QtGui.QCheckBox(self.fr_reg_pipe_1)
        self.cb_prealign_images.setEnabled(False)
        self.cb_prealign_images.setGeometry(QtCore.QRect(10, 80, 141, 22))
        self.cb_prealign_images.setObjectName("cb_prealign_images")
        self.fr_reg_pipe_2 = QtGui.QFrame(self.tab_reg_settings)
        self.fr_reg_pipe_2.setGeometry(QtCore.QRect(230, 340, 161, 131))
        self.fr_reg_pipe_2.setFrameShape(QtGui.QFrame.StyledPanel)
        self.fr_reg_pipe_2.setFrameShadow(QtGui.QFrame.Raised)
        self.fr_reg_pipe_2.setObjectName("fr_reg_pipe_2")
        self.cb_affine_registration = QtGui.QCheckBox(self.fr_reg_pipe_2)
        self.cb_affine_registration.setGeometry(QtCore.QRect(10, 10, 141, 22))
        self.cb_affine_registration.setObjectName("cb_affine_registration")
        self.l_max_iterations_p2 = QtGui.QLabel(self.fr_reg_pipe_2)
        self.l_max_iterations_p2.setGeometry(QtCore.QRect(10, 30, 141, 21))
        self.l_max_iterations_p2.setObjectName("l_max_iterations_p2")
        self.sb_max_iterations_p2 = QtGui.QSpinBox(self.fr_reg_pipe_2)
        self.sb_max_iterations_p2.setEnabled(False)
        self.sb_max_iterations_p2.setGeometry(QtCore.QRect(10, 50, 141, 21))
        self.sb_max_iterations_p2.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.sb_max_iterations_p2.setMaximum(1000)
        self.sb_max_iterations_p2.setProperty("value", 20)
        self.sb_max_iterations_p2.setObjectName("sb_max_iterations_p2")
        self.fr_reg_pipe_3 = QtGui.QFrame(self.tab_reg_settings)
        self.fr_reg_pipe_3.setGeometry(QtCore.QRect(430, 340, 191, 131))
        self.fr_reg_pipe_3.setFrameShape(QtGui.QFrame.StyledPanel)
        self.fr_reg_pipe_3.setFrameShadow(QtGui.QFrame.Raised)
        self.fr_reg_pipe_3.setObjectName("fr_reg_pipe_3")
        self.cb_deformable_registration = QtGui.QCheckBox(self.fr_reg_pipe_3)
        self.cb_deformable_registration.setGeometry(QtCore.QRect(10, 10, 181, 22))
        self.cb_deformable_registration.setObjectName("cb_deformable_registration")
        self.l_max_iterations_p3 = QtGui.QLabel(self.fr_reg_pipe_3)
        self.l_max_iterations_p3.setGeometry(QtCore.QRect(10, 31, 141, 20))
        self.l_max_iterations_p3.setObjectName("l_max_iterations_p3")
        self.sb_max_iterations_p3 = QtGui.QSpinBox(self.fr_reg_pipe_3)
        self.sb_max_iterations_p3.setEnabled(False)
        self.sb_max_iterations_p3.setGeometry(QtCore.QRect(10, 51, 141, 20))
        self.sb_max_iterations_p3.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.sb_max_iterations_p3.setMaximum(1000)
        self.sb_max_iterations_p3.setProperty("value", 20)
        self.sb_max_iterations_p3.setObjectName("sb_max_iterations_p3")
        self.sb_max_deformation = QtGui.QSpinBox(self.fr_reg_pipe_3)
        self.sb_max_deformation.setEnabled(False)
        self.sb_max_deformation.setGeometry(QtCore.QRect(10, 100, 141, 21))
        self.sb_max_deformation.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.sb_max_deformation.setMaximum(1000)
        self.sb_max_deformation.setProperty("value", 10)
        self.sb_max_deformation.setObjectName("sb_max_deformation")
        self.l_max_deformation = QtGui.QLabel(self.fr_reg_pipe_3)
        self.l_max_deformation.setGeometry(QtCore.QRect(10, 80, 161, 16))
        self.l_max_deformation.setObjectName("l_max_deformation")
        self.l_max_deformation_mm = QtGui.QLabel(self.fr_reg_pipe_3)
        self.l_max_deformation_mm.setGeometry(QtCore.QRect(160, 100, 31, 21))
        self.l_max_deformation_mm.setObjectName("l_max_deformation_mm")
        self.l_next1 = QtGui.QLabel(self.tab_reg_settings)
        self.l_next1.setGeometry(QtCore.QRect(190, 340, 31, 131))
        self.l_next1.setObjectName("l_next1")
        self.l_next2 = QtGui.QLabel(self.tab_reg_settings)
        self.l_next2.setGeometry(QtCore.QRect(400, 340, 31, 131))
        self.l_next2.setObjectName("l_next2")
        self.tabWidget.addTab(self.tab_reg_settings, "")
        self.tab_help = QtGui.QWidget()
        self.tab_help.setObjectName("tab_help")
        self.comboBox_help = QtGui.QComboBox(self.tab_help)
        self.comboBox_help.setGeometry(QtCore.QRect(10, 10, 241, 21))
        self.comboBox_help.setObjectName("comboBox_help")
        self.textBrowser_help = QtGui.QTextBrowser(self.tab_help)
        self.textBrowser_help.setGeometry(QtCore.QRect(10, 40, 621, 471))
        self.textBrowser_help.setObjectName("textBrowser_help")
        self.tabWidget.addTab(self.tab_help, "")
        self.progressBar = QtGui.QProgressBar(Fenster)
        self.progressBar.setGeometry(QtCore.QRect(10, 680, 651, 23))
        self.progressBar.setProperty("value", 1)
        self.progressBar.setFormat("")
        self.progressBar.setObjectName("progressBar")
        self.b_start_proc = QtGui.QPushButton(Fenster)
        self.b_start_proc.setGeometry(QtCore.QRect(10, 600, 321, 27))
        self.b_start_proc.setObjectName("b_start_proc")
        self.b_save_settings = QtGui.QPushButton(Fenster)
        self.b_save_settings.setGeometry(QtCore.QRect(340, 600, 321, 27))
        self.b_save_settings.setObjectName("b_save_settings")
        self.b_exit = QtGui.QPushButton(Fenster)
        self.b_exit.setGeometry(QtCore.QRect(10, 640, 321, 27))
        self.b_exit.setObjectName("b_exit")
        self.b_load_settings = QtGui.QPushButton(Fenster)
        self.b_load_settings.setGeometry(QtCore.QRect(340, 640, 321, 27))
        self.b_load_settings.setObjectName("b_load_settings")

        self.retranslateUi(Fenster)
        self.tabWidget.setCurrentIndex(1)
        QtCore.QObject.connect(self.b_exit, QtCore.SIGNAL("clicked()"), Fenster.close)
        QtCore.QObject.connect(self.cb_atlas_registration, QtCore.SIGNAL("clicked(bool)"), self.cb_show_registration_results.setEnabled)
        QtCore.QObject.connect(self.cb_create_mask, QtCore.SIGNAL("clicked(bool)"), self.cb_show_mask.setEnabled)
        QtCore.QObject.connect(self.cb_set_repetition, QtCore.SIGNAL("clicked(bool)"), self.dsb_repetition_time.setEnabled)
        QtCore.QObject.connect(self.cb_high_pass, QtCore.SIGNAL("clicked(bool)"), self.sb_hp_cutoff.setEnabled)
        QtCore.QObject.connect(self.cb_low_pass, QtCore.SIGNAL("clicked(bool)"), self.sb_lp_cutoff.setEnabled)
        QtCore.QObject.connect(self.cb_rigid_registration, QtCore.SIGNAL("clicked(bool)"), self.l_max_iterations_p1.setEnabled)
        QtCore.QObject.connect(self.cb_rigid_registration, QtCore.SIGNAL("clicked(bool)"), self.sb_max_iterations_p1.setEnabled)
        QtCore.QObject.connect(self.cb_rigid_registration, QtCore.SIGNAL("clicked(bool)"), self.cb_prealign_images.setEnabled)
        QtCore.QObject.connect(self.cb_affine_registration, QtCore.SIGNAL("clicked(bool)"), self.sb_max_iterations_p2.setEnabled)
        QtCore.QObject.connect(self.cb_deformable_registration, QtCore.SIGNAL("clicked(bool)"), self.sb_max_iterations_p3.setEnabled)
        QtCore.QObject.connect(self.cb_deformable_registration, QtCore.SIGNAL("clicked(bool)"), self.sb_max_deformation.setEnabled)
        QtCore.QMetaObject.connectSlotsByName(Fenster)

    def retranslateUi(self, Fenster):
        Fenster.setWindowTitle(QtGui.QApplication.translate("Fenster", "Lipsia Preprocessing", None, QtGui.QApplication.UnicodeUTF8))
        self.l_cutoff_lp_s.setText(QtGui.QApplication.translate("Fenster", "s", None, QtGui.QApplication.UnicodeUTF8))
        self.cb_spatial_filtering.setText(QtGui.QApplication.translate("Fenster", "Spatial filtering", None, QtGui.QApplication.UnicodeUTF8))
        self.b_add_image_directory.setText(QtGui.QApplication.translate("Fenster", "Add dicom dir", None, QtGui.QApplication.UnicodeUTF8))
        self.cb_debug_output.setText(QtGui.QApplication.translate("Fenster", "Debug Output", None, QtGui.QApplication.UnicodeUTF8))
        self.cb_slicetime_correction.setText(QtGui.QApplication.translate("Fenster", "Slicetime correction", None, QtGui.QApplication.UnicodeUTF8))
        self.l_preprocessing_steps.setText(QtGui.QApplication.translate("Fenster", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Sans\'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:9pt; font-weight:600;\">Preprocessing steps</span></p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.l_cutoff_high_pass.setText(QtGui.QApplication.translate("Fenster", "Cutoff:", None, QtGui.QApplication.UnicodeUTF8))
        self.cb_set_repetition.setText(QtGui.QApplication.translate("Fenster", "Set repetition time:", None, QtGui.QApplication.UnicodeUTF8))
        self.l_input_files.setText(QtGui.QApplication.translate("Fenster", "Input fIles:", None, QtGui.QApplication.UnicodeUTF8))
        self.l_mask_options.setText(QtGui.QApplication.translate("Fenster", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Sans\'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:9pt; font-weight:600;\">Mask options</span></p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.cb_low_pass.setText(QtGui.QApplication.translate("Fenster", "Low pass", None, QtGui.QApplication.UnicodeUTF8))
        self.cb_fieldmap_correction.setText(QtGui.QApplication.translate("Fenster", "Fieldmap correction", None, QtGui.QApplication.UnicodeUTF8))
        self.cb_write_logfile.setText(QtGui.QApplication.translate("Fenster", "Write logfile", None, QtGui.QApplication.UnicodeUTF8))
        self.l_cutoff_low_pass.setText(QtGui.QApplication.translate("Fenster", "Cutoff:", None, QtGui.QApplication.UnicodeUTF8))
        self.l_miscellaneous.setText(QtGui.QApplication.translate("Fenster", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Sans\'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:9pt; font-weight:600;\">Miscellaneous</span></p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.l_spatial_filtering.setText(QtGui.QApplication.translate("Fenster", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Sans\'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:9pt; font-weight:600;\">Spatial filtering</span></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:9pt; font-weight:600;\"></p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.l_spatial_filtering_mm.setText(QtGui.QApplication.translate("Fenster", "mm", None, QtGui.QApplication.UnicodeUTF8))
        self.cb_atlas_registration.setText(QtGui.QApplication.translate("Fenster", "Atlas registration", None, QtGui.QApplication.UnicodeUTF8))
        self.l_output.setText(QtGui.QApplication.translate("Fenster", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Sans\'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">Output</span></p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.cb_show_mask.setText(QtGui.QApplication.translate("Fenster", "Show mask (interruption)", None, QtGui.QApplication.UnicodeUTF8))
        self.l_fwhm.setText(QtGui.QApplication.translate("Fenster", "FWHM:", None, QtGui.QApplication.UnicodeUTF8))
        self.cb_show_registration_results.setText(QtGui.QApplication.translate("Fenster", "Show registration results (interruption)", None, QtGui.QApplication.UnicodeUTF8))
        self.cb_create_mask.setText(QtGui.QApplication.translate("Fenster", "Create a mask", None, QtGui.QApplication.UnicodeUTF8))
        self.b_add_image_file.setText(QtGui.QApplication.translate("Fenster", "Add images", None, QtGui.QApplication.UnicodeUTF8))
        self.b_remove_file.setText(QtGui.QApplication.translate("Fenster", "Remove", None, QtGui.QApplication.UnicodeUTF8))
        self.l_cutoff_hp_s.setText(QtGui.QApplication.translate("Fenster", "s", None, QtGui.QApplication.UnicodeUTF8))
        self.cb_movement_correction.setText(QtGui.QApplication.translate("Fenster", "Movement correction", None, QtGui.QApplication.UnicodeUTF8))
        self.l_directory.setText(QtGui.QApplication.translate("Fenster", "Directory:", None, QtGui.QApplication.UnicodeUTF8))
        self.l_temporal_filtering.setText(QtGui.QApplication.translate("Fenster", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Sans\'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:9pt; font-weight:600;\">Temporal filtering</span></p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.cb_high_pass.setText(QtGui.QApplication.translate("Fenster", "High pass filter", None, QtGui.QApplication.UnicodeUTF8))
        self.cb_temporal_filtering.setText(QtGui.QApplication.translate("Fenster", "Temporal filtering", None, QtGui.QApplication.UnicodeUTF8))
        self.le_directory.setText(QtGui.QApplication.translate("Fenster", "/home/blah/blah", None, QtGui.QApplication.UnicodeUTF8))
        self.le_prefix.setText(QtGui.QApplication.translate("Fenster", "preproc_", None, QtGui.QApplication.UnicodeUTF8))
        self.l_prefix.setText(QtGui.QApplication.translate("Fenster", "Prefix", None, QtGui.QApplication.UnicodeUTF8))
        self.rb_create_average_mask.setText(QtGui.QApplication.translate("Fenster", "Create an average mask", None, QtGui.QApplication.UnicodeUTF8))
        self.rb_create_mask_subject.setText(QtGui.QApplication.translate("Fenster", "Create a mask for every subject", None, QtGui.QApplication.UnicodeUTF8))
        self.l_maximum_number_of_voxel.setText(QtGui.QApplication.translate("Fenster", "Maximum number of voxel:", None, QtGui.QApplication.UnicodeUTF8))
        self.b_output_directory.setText(QtGui.QApplication.translate("Fenster", "...", None, QtGui.QApplication.UnicodeUTF8))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_general), QtGui.QApplication.translate("Fenster", "General", None, QtGui.QApplication.UnicodeUTF8))
        self.l_atlas_registration.setText(QtGui.QApplication.translate("Fenster", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Sans\'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:9pt; font-weight:600;\">Atlas registration</span></p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.l_template.setText(QtGui.QApplication.translate("Fenster", "Template:", None, QtGui.QApplication.UnicodeUTF8))
        self.b_view.setText(QtGui.QApplication.translate("Fenster", "View", None, QtGui.QApplication.UnicodeUTF8))
        self.l_functional_anatomical_image.setText(QtGui.QApplication.translate("Fenster", "Functional image -> Anatomical image", None, QtGui.QApplication.UnicodeUTF8))
        self.b_remove_anatomical_image.setText(QtGui.QApplication.translate("Fenster", "Remove anatomical image", None, QtGui.QApplication.UnicodeUTF8))
        self.b_add_anatomical_image.setText(QtGui.QApplication.translate("Fenster", "Add anatomical image", None, QtGui.QApplication.UnicodeUTF8))
        self.l_registration_pipeline.setText(QtGui.QApplication.translate("Fenster", "Registration pipeline:", None, QtGui.QApplication.UnicodeUTF8))
        self.l_interpolator_function.setText(QtGui.QApplication.translate("Fenster", "Interpolator function:", None, QtGui.QApplication.UnicodeUTF8))
        self.l_output_resolution.setText(QtGui.QApplication.translate("Fenster", "Output resolution:", None, QtGui.QApplication.UnicodeUTF8))
        self.le_output_resolution.setText(QtGui.QApplication.translate("Fenster", "3,3,3", None, QtGui.QApplication.UnicodeUTF8))
        self.cb_rigid_registration.setText(QtGui.QApplication.translate("Fenster", "Rigid Registration", None, QtGui.QApplication.UnicodeUTF8))
        self.l_max_iterations_p1.setText(QtGui.QApplication.translate("Fenster", "Maximum iterations:", None, QtGui.QApplication.UnicodeUTF8))
        self.cb_prealign_images.setText(QtGui.QApplication.translate("Fenster", "Prealign images", None, QtGui.QApplication.UnicodeUTF8))
        self.cb_affine_registration.setText(QtGui.QApplication.translate("Fenster", "Affine Registration", None, QtGui.QApplication.UnicodeUTF8))
        self.l_max_iterations_p2.setText(QtGui.QApplication.translate("Fenster", "Maximum iterations:", None, QtGui.QApplication.UnicodeUTF8))
        self.cb_deformable_registration.setText(QtGui.QApplication.translate("Fenster", "Deformable registration", None, QtGui.QApplication.UnicodeUTF8))
        self.l_max_iterations_p3.setText(QtGui.QApplication.translate("Fenster", "Maximum iterations:", None, QtGui.QApplication.UnicodeUTF8))
        self.l_max_deformation.setText(QtGui.QApplication.translate("Fenster", "Maximum deformation:", None, QtGui.QApplication.UnicodeUTF8))
        self.l_max_deformation_mm.setText(QtGui.QApplication.translate("Fenster", "mm", None, QtGui.QApplication.UnicodeUTF8))
        self.l_next1.setText(QtGui.QApplication.translate("Fenster", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Sans\'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:24pt; font-weight:600;\">&gt;</span></p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.l_next2.setText(QtGui.QApplication.translate("Fenster", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'Sans\'; font-size:10pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:24pt; font-weight:600;\">&gt;</span></p></body></html>", None, QtGui.QApplication.UnicodeUTF8))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_reg_settings), QtGui.QApplication.translate("Fenster", "Registration Settings", None, QtGui.QApplication.UnicodeUTF8))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_help), QtGui.QApplication.translate("Fenster", "Help", None, QtGui.QApplication.UnicodeUTF8))
        self.b_start_proc.setText(QtGui.QApplication.translate("Fenster", "Start preprocessing", None, QtGui.QApplication.UnicodeUTF8))
        self.b_save_settings.setText(QtGui.QApplication.translate("Fenster", "Save Settings", None, QtGui.QApplication.UnicodeUTF8))
        self.b_exit.setText(QtGui.QApplication.translate("Fenster", "Exit", None, QtGui.QApplication.UnicodeUTF8))
        self.b_load_settings.setText(QtGui.QApplication.translate("Fenster", "Load  Settings", None, QtGui.QApplication.UnicodeUTF8))

