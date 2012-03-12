#
#pregui.py
#
import sys
import string
import os
from PyQt4 import QtCore, QtGui, QtWebKit
import ConfigParser

#Gui-File importieren
from vpreproc_gui import Ui_Fenster



class PreProc(QtGui.QMainWindow):
    def __init__(self, app, parent = None):
        QtGui.QMainWindow.__init__(self,parent)
        self.app = app
        self.ui = Ui_Fenster()
        self.ui.setupUi(self)

        self.ui.setupUi(self)
        self.homePath = QtCore.QDir.currentPath()
        self.myFileList = []
        self.currentImageIndex = 0
        self.removeList = []
        self.removeListEver = []
        self.anatomicalImageDict = {}
        self.myProcessList = []
        self.showDebugOutput = False
        self.ui.le_directory.setText(self.homePath)
        self.output_dir=self.homePath        
        self.flag_atlas_on = True

        self.statusBar().showMessage("Welcome!")
        self.templateDict = {"T1 MNI 1mm fullbrain": "/usr/share/lipsia/mni.v",
                             "EPI 2mm": "/usr/share/lipsia/epi_template_2mm.v",
                             "EPI 3mm":"/usr/share/lipsia/epi_template_3mm.v",         
                             "EPI 1.5mm":"/usr/share/lipsia/epi_template_1.5mm.v"
                            }
        for template, path in self.templateDict.iteritems():
            if(os.path.isfile(path)):
                self.ui.comboBox_template.addItem(template)
        self.ui.comboBox_template.setCurrentIndex(2) 
        self.helpDict = {"Lipsia help": "/usr/share/doc/lipsia/html/index.html" }
        for help, path in self.helpDict.iteritems():
            if(os.path.isfile(path)):
                self.ui.comboBox_help.addItem(help)
        self.show_help()
        self.interpolators = ["Linear", "BSpline", "NearestNeighbor"]
        for inter in self.interpolators:
            self.ui.comboBox_interpol_func.addItem(inter) 

        #list connections
        QtCore.QObject.connect(self.ui.b_add_image_file, QtCore.SIGNAL("clicked()"), self.add_data_to_list )
        QtCore.QObject.connect(self.ui.b_add_image_directory, QtCore.SIGNAL("clicked()"), self.add_dir_to_list )
        QtCore.QObject.connect(self.ui.b_remove_file, QtCore.SIGNAL("clicked()"), self.remove_from_list )
        QtCore.QObject.connect(self.ui.b_start_proc, QtCore.SIGNAL("clicked()"), self.start_processing )
        QtCore.QObject.connect(self.ui.b_output_directory, QtCore.SIGNAL("clicked()"), self.open_output_dir )
        QtCore.QObject.connect(self.ui.b_add_anatomical_image, QtCore.SIGNAL("clicked()"), self.add_anatomical_image )
        QtCore.QObject.connect(self.ui.b_view, QtCore.SIGNAL("clicked()"), self.view_template_image )
        QtCore.QObject.connect(self.ui.b_remove_anatomical_image, QtCore.SIGNAL("clicked()"), self.remove_anatomical_image )
        QtCore.QObject.connect(self.ui.b_save_settings, QtCore.SIGNAL("clicked()"), self.save_settings_to_file )
        QtCore.QObject.connect(self.ui.b_load_settings, QtCore.SIGNAL("clicked()"), self.load_settings )
        QtCore.QObject.connect(self.ui.comboBox_help, QtCore.SIGNAL("currentIndexChanged(int)"), self.show_help )
        QtCore.QObject.connect(self.ui.cb_atlas_registration, QtCore.SIGNAL("clicked()"), self.change_atlas_reg)
        QtCore.QObject.connect(self.ui.le_directory, QtCore.SIGNAL("editingFinishes()"), self.change_output_directory)

#Einzelnes File hinzufuegen
    def add_data_to_list(self):
        dataList=QtGui.QFileDialog.getOpenFileNames(self, "Select images", self.homePath, ("Images (*.nii *.v)"))
        if(len(dataList)):
            for fileName in dataList:
                self.ui.lw_input_files.addItem(fileName)
                self.ui.lw_functional_anatomical_image.addItem(fileName + " -> ")
                self.myFileList.append(fileName)

#Verzeichnis hinzufuegen
    def add_dir_to_list(self):
        dataDir=QtGui.QFileDialog.getExistingDirectory(self, "Data directory", self.homePath)
        if(len(dataDir)):
            self.ui.lw_input_files.addItem(dataDir)
            self.ui.lw_functional_anatomical_image.addItem(dataDir + " -> ")
            self.myFileList.append(dataDir)

#Aus Liste entfernen
    def remove_from_list(self):
        numberSelect = self.ui.lw_input_files.currentRow()
        self.myFileList.remove(self.ui.lw_input_files.item(numberSelect).text())
        try:
            del self.anatomicalImageDict[str(self.ui.lw_input_files.item(numberSelect).text())]
        except:
            pass
        self.ui.lw_input_files.takeItem(numberSelect)
        self.ui.lw_functional_anatomical_image.takeItem(numberSelect)


#Ausgabeverzeichnis waehlen
    def open_output_dir(self):
        self.output_dir=QtGui.QFileDialog.getExistingDirectory(self, "Ouput directory", self.homePath)
        self.ui.le_directory.setText(self.output_dir)

#Anatomisches Bild hinzufuegen
    def add_anatomical_image(self):
        row = self.ui.lw_functional_anatomical_image.currentRow()
        if (not row == -1):
            anaImage=QtGui.QFileDialog.getOpenFileName(self, "Select anatomical image", self.homePath, ("Image (*.nii *.v)"))
            if(len(anaImage)):
                funcFile = (str(self.ui.lw_functional_anatomical_image.item(row).text()).split(" -> "))[0]
                self.anatomicalImageDict[funcFile] = anaImage    
                listString = funcFile + " -> " + anaImage
                self.ui.lw_functional_anatomical_image.takeItem(row)
                self.ui.lw_functional_anatomical_image.insertItem(row, listString)
                self.ui.comboBox_template.setCurrentIndex(0)

#Zeigt ausgewaehltes Templateimage an
    def view_template_image(self):
        imageToShow = self.templateDict[str(self.ui.comboBox_template.currentText())]
        os.system("vlv -in " +  imageToShow + " 2> tmpOut")
        os.remove("tmpOut")

#Anatomisches Bild entfernen
    def remove_anatomical_image(self):
        row = self.ui.lw_functional_anatomical_image.currentRow()
        if (not row == -1):
            listString = str(self.ui.lw_functional_anatomical_image.item(row).text())
            funcImage = listString.split(" -> ")[0] + " -> "
            self.ui.lw_functional_anatomical_image.takeItem(row)
            self.ui.lw_functional_anatomical_image.insertItem(row, funcImage)
            del self.anatomicalImageDict[funcImage.rstrip(" -> ")]
            if(not len(self.anatomicalImageDict) ):
                self.ui.comboBox_template.setCurrentIndex(3)

#ConfigParser zum Speichern
    def save_settings_to_file(self):
        
        config = ConfigParser.RawConfigParser()
        config.add_section('Preprocessing Steps')
        config.set('Preprocessing Steps', 'atlas_registration', self.ui.cb_atlas_registration.isChecked())
        config.set('Preprocessing Steps', 'create_mask', self.ui.cb_create_mask.isChecked())
        config.set('Preprocessing Steps', 'debug_output', self.ui.cb_debug_output.isChecked())
        config.set('Preprocessing Steps', 'fieldmap_correction', self.ui.cb_fieldmap_correction.isChecked())
        config.set('Preprocessing Steps', 'high_pass', self.ui.cb_high_pass.isChecked())
        config.set('Preprocessing Steps', 'low_pass', self.ui.cb_low_pass.isChecked())
        config.set('Preprocessing Steps', 'movement_correction', self.ui.cb_movement_correction.isChecked())
        config.set('Preprocessing Steps', 'set_repetition', self.ui.cb_set_repetition.isChecked())
        config.set('Preprocessing Steps', 'show_mask', self.ui.cb_show_mask.isChecked())
        config.set('Preprocessing Steps', 'show_registration_results', self.ui.cb_show_registration_results.isChecked())
        config.set('Preprocessing Steps', 'slicetime_correction', self.ui.cb_slicetime_correction.isChecked())
        config.set('Preprocessing Steps', 'spatial_filtering', self.ui.cb_spatial_filtering.isChecked())
        config.set('Preprocessing Steps', 'temporal_filtering', self.ui.cb_temporal_filtering.isChecked())
        config.set('Preprocessing Steps', 'write_logfile', self.ui.cb_write_logfile.isChecked())
        config.set('Preprocessing Steps', 'FWHM in mm', self.ui.sb_fwhm.value())
        config.set('Preprocessing Steps', 'High Pass Filter Cutoff in s', self.ui.sb_hp_cutoff.value())
        config.set('Preprocessing Steps', 'Low Pass Filter Cutoff in s', self.ui.sb_lp_cutoff.value())
        config.set('Preprocessing Steps', 'Maximum number of voxel', self.ui.sb_max_num_voxel.value())
        config.set('Preprocessing Steps', 'create_average_mask', self.ui.rb_create_average_mask.isChecked())
        config.set('Preprocessing Steps', 'create_mask_subject', self.ui.rb_create_mask_subject.isChecked())
        config.set('Preprocessing Steps', 'repetition_time', self.ui.dsb_repetition_time.value())
        config.set('Preprocessing Steps', 'output_directory', self.ui.le_directory.text())
        config.set('Preprocessing Steps', 'prefix', self.ui.le_prefix.text())

        config.add_section('Registration Settings')
        config.set('Registration Settings', 'Atlas Template number', self.ui.comboBox_template.currentIndex())
        config.set('Registration Settings', 'Atlas Template text', self.ui.comboBox_template.currentText())
        config.set('Registration Settings', 'ridid_registration', self.ui.cb_rigid_registration.isChecked())
        config.set('Registration Settings', 'max_iterations_p1', self.ui.sb_max_iterations_p1.value())
        config.set('Registration Settings', 'prealing_images', self.ui.cb_prealign_images.isChecked())
        config.set('Registration Settings', 'affine_registration', self.ui.cb_affine_registration.isChecked())
        config.set('Registration Settings', 'max_iterations_p2', self.ui.sb_max_iterations_p2.value())
        config.set('Registration Settings', 'deformable_registration', self.ui.cb_deformable_registration.isChecked())
        config.set('Registration Settings', 'max_iterations_p3', self.ui.sb_max_iterations_p3.value())
        config.set('Registration Settings', 'max_deformation in mm', self.ui.sb_max_deformation.value())
        config.set('Registration Settings', 'Interpolator function', self.ui.comboBox_interpol_func.currentIndex())
        config.set('Registration Settings', 'Output resolution', self.ui.le_output_resolution.text())

        configFileName = QtGui.QFileDialog.getSaveFileName(self, "Save settings to file", self.homePath, ("*.lpp"))
        regExp = QtCore.QRegExp(QtCore.QString("*.lpp"))
        regExp.setPatternSyntax(QtCore.QRegExp.Wildcard)
        if(not configFileName.isEmpty()):
            if not regExp.exactMatch(configFileName):
                configFileName.append('.lpp')
            with open(configFileName, 'wb') as configfile:
                config.write(configfile)
        else:
           self.throwError("Please specify a filename")
        
#...und laden der Einstellungen
    def load_settings(self):
        config = ConfigParser.RawConfigParser()
        configFileName = QtGui.QFileDialog.getOpenFileName(self, "Load settings from file", self.homePath, ("*.lpp"))
        if(not configFileName.isEmpty()):
            config.readfp(open(configFileName))
        else:
            self.throwError("Please specify a filename")
        self.ui.cb_atlas_registration.setChecked(config.getboolean('Preprocessing Steps', 'atlas_registration'))
        self.change_atlas_reg()
        self.ui.cb_create_mask.setChecked(config.getboolean('Preprocessing Steps', 'create_mask'))
        self.ui.cb_debug_output.setChecked(config.getboolean('Preprocessing Steps', 'debug_output'))
        self.ui.cb_fieldmap_correction.setChecked(config.getboolean('Preprocessing Steps', 'fieldmap_correction'))
        self.ui.cb_high_pass.setChecked(config.getboolean('Preprocessing Steps', 'high_pass'))
        self.ui.cb_low_pass.setChecked(config.getboolean('Preprocessing Steps', 'low_pass'))
        self.ui.cb_movement_correction.setChecked(config.getboolean('Preprocessing Steps', 'movement_correction'))
        self.ui.cb_set_repetition.setChecked(config.getboolean('Preprocessing Steps', 'set_repetition'))
        self.ui.cb_show_mask.setEnabled(self.ui.cb_create_mask.isChecked())
        self.ui.cb_show_mask.setChecked(config.getboolean('Preprocessing Steps', 'show_mask'))
        self.ui.cb_show_registration_results.setEnabled(self.ui.cb_atlas_registration.isChecked())
        if self.flag_atlas_on:
            self.ui.cb_show_registration_results.setChecked(config.getboolean('Preprocessing Steps', 'show_registration_results'))
        self.ui.cb_slicetime_correction.setChecked(config.getboolean('Preprocessing Steps', 'slicetime_correction'))
        self.ui.cb_spatial_filtering.setChecked(config.getboolean('Preprocessing Steps', 'spatial_filtering'))
        self.ui.cb_temporal_filtering.setChecked(config.getboolean('Preprocessing Steps', 'temporal_filtering'))
        self.ui.cb_write_logfile.setChecked(config.getboolean('Preprocessing Steps', 'write_logfile'))
        self.ui.sb_fwhm.setValue(config.getint('Preprocessing Steps', 'FWHM in mm'))
        self.ui.sb_hp_cutoff.setEnabled(self.ui.cb_high_pass.isChecked())
        self.ui.sb_hp_cutoff.setValue(config.getint('Preprocessing Steps', 'High Pass Filter Cutoff in s'))
        self.ui.sb_lp_cutoff.setEnabled(self.ui.cb_low_pass.isChecked())
        self.ui.sb_lp_cutoff.setValue(config.getint('Preprocessing Steps', 'Low Pass Filter Cutoff in s'))
        self.ui.sb_max_num_voxel.setValue(config.getint('Preprocessing Steps', 'Maximum number of voxel'))
        self.ui.rb_create_average_mask.setChecked(config.getboolean('Preprocessing Steps', 'create_average_mask'))
        self.ui.rb_create_mask_subject.setChecked(config.getboolean('Preprocessing Steps', 'create_mask_subject'))
        self.ui.dsb_repetition_time.setEnabled(self.ui.cb_set_repetition.isChecked())
        self.ui.dsb_repetition_time.setValue(config.getfloat('Preprocessing Steps', 'repetition_time'))
        self.ui.le_directory.setText(config.get('Preprocessing Steps', 'output_directory'))
        self.change_output_directory()
        self.ui.le_prefix.setText(config.get('Preprocessing Steps', 'prefix'))


        template_number = config.getint('Registration Settings', 'Atlas Template number')
        if self.ui.comboBox_template.itemText(config.getint('Registration Settings', 'Atlas Template number')) == config.get('Registration Settings', 'Atlas Template text'):
            self.ui.comboBox_template.setCurrentIndex(template_number)
        else:
           self.throwError("Saved template not available")
        self.ui.cb_rigid_registration.setChecked(config.getboolean('Registration Settings', 'ridid_registration'))
        self.ui.sb_max_iterations_p1.setEnabled(self.ui.cb_rigid_registration.isChecked())
        self.ui.sb_max_iterations_p1.setValue(config.getint('Registration Settings', 'max_iterations_p1'))
        self.ui.cb_prealign_images.setEnabled(self.ui.cb_rigid_registration.isChecked())
        self.ui.cb_prealign_images.setChecked(config.getboolean('Registration Settings', 'prealing_images'))
        self.ui.cb_affine_registration.setChecked(config.getboolean('Registration Settings', 'affine_registration'))
        self.ui.sb_max_iterations_p2.setEnabled(self.ui.cb_affine_registration.isChecked())
        self.ui.sb_max_iterations_p2.setValue(config.getint('Registration Settings', 'max_iterations_p2'))
        self.ui.cb_deformable_registration.setChecked(config.getboolean('Registration Settings', 'deformable_registration'))
        self.ui.sb_max_iterations_p3.setEnabled(self.ui.cb_deformable_registration.isChecked())
        self.ui.sb_max_iterations_p3.setValue(config.getint('Registration Settings', 'max_iterations_p3'))
        self.ui.sb_max_deformation.setEnabled(self.ui.cb_deformable_registration.isChecked())
        self.ui.sb_max_deformation.setValue(config.getint('Registration Settings', 'max_deformation in mm'))
        self.ui.comboBox_interpol_func.setCurrentIndex(config.getint('Registration Settings', 'Interpolator function'))
        self.ui.le_output_resolution.setText(config.get('Registration Settings', 'Output resolution'))

#help fehlt
    def show_help(self):
        if( self.ui.comboBox_help.currentText() != "" ):
            self.ui.textBrowser_help.setUrl(QtCore.QUrl(self.helpDict[str(self.ui.comboBox_help.currentText())]))

# Template auswahlcheck - wenn die nicht vorhanden sind (lipsia-paket), dann fehler schmeissen und wieder ausschalten
    def change_atlas_reg(self):
        if(self.ui.cb_atlas_registration.isChecked()):
            if(not self.ui.comboBox_template.count()):
                self.throwError("No template image was found. Please make sure you have installed the lipsia-sandbox package!")
                self.ui.cb_atlas_registration.setChecked(False)
                self.ui.cb_show_registration_results.setChecked(False)
                self.ui.cb_show_registration_results.setEnabled(False)
                self.flag_atlas_on = False
            else:
                self.flag_atlas_on = True
#Sicherstellen, dass die output_dir - variable mit dem uebereinstimmt was im Textfeld steht
    def change_output_directory(self):
        self.output_dir = self.ui.le_directory.text()

##################################################################################################################################

#Verarbeitung starten
    def start_processing(self):
        if(len(self.myFileList)):
            self.calculateProgressBar()
            self.ui.progressBar.setValue(0)
            self.ui.b_start_proc.setEnabled(False)
            for myFile in self.myFileList:
                if (self.convertAndCheck(str(myFile))):
                    checked = True
                else:
                    checked = False
                    break
            if (checked):
                finalList=[]
                for file in self.myProcessList:
                    origName = file
                    if(self.ui.cb_slicetime_correction.isChecked()):
                        file = self.doSliceTimeCorrection(file)
                    if(self.ui.cb_movement_correction.isChecked()):
                        file = self.doMovementCorrection(file)
                    if(self.ui.cb_atlas_registration.isChecked()):
                        file = self.doRegistration(file)
                    if(self.ui.cb_spatial_filtering.isChecked()):
                        file = self.doSpatialFiltering(file)
                    if(self.ui.cb_temporal_filtering.isChecked()):
                        file = self.doTemporalFiltering(file)
                    if(self.ui.cb_create_mask.isChecked() and self.ui.rb_create_mask_subject.isChecked()):
                        self.doCreateMaskFromFile(file, origName)
                    finalList.append(file)
                    outFile = str(self.output_dir).rstrip("/") + "/" + self.ui.le_prefix.text() + origName.split("/")[len(origName.split("/"))-1]
                    self.applyCommand("cp " + file + " " + outFile)
                if(self.ui.cb_create_mask.isChecked() and self.ui.rb_create_average_mask.isChecked()):
                    self.doCreateMaskAverage(finalList)
                        
                self.currentImageIndex += 1
            else:
                print "Processing stopped!"
            self.ui.progressBar.setValue(0)
            self.progress()
            self.currentImageIndex = 0
            self.debugOutput("Done!", True)
            self.ui.b_start_proc.setEnabled(True)        

    def convertAndCheck(self, myFile):
        myFile = self.convertFiles(myFile)
        if(self.checkAll(myFile)):
            self.myProcessList.append(myFile)
            return True
        else:
            return False

    def convertFiles(self, myFile):
        self.progress()
        readFormat=""
        if( string.find(myFile, ".v") == -1 and string.find(myFile, ".nii") == -1 and string.find(myFile, ".ima") == -1 and string.find(myFile, ".dcm") == -1 ):
            readFormat = "-rf .ima"
        if( string.find(myFile, ".v") == -1 or self.ui.cb_set_repetition.isChecked() ):
            self.debugOutput("Converting " + myFile + " to vista...", True)
            outFile = myFile.rstrip("/")
            outFile = outFile.rstrip(".nii")
            outFile = outFile.rstrip(".v")
            outFile = outFile.split("/")[len(outFile.split("/"))-1]
            #outFile = "conv_" + outFile
            if(self.ui.cb_set_repetition.isChecked()):
                tr = self.ui.dsb_repetition_time.value()
                self.applyCommand("vvinidi -in " + myFile + " -out " + str(self.output_dir).rstrip("/") + "/" + outFile + ".v -tr " + str(tr) + " " + readFormat)
            else:
                self.applyCommand("vvinidi -in " + myFile + " -out " + str(self.output_dir).rstrip("/") + "/" + outFile + ".v " + readFormat)
            self.removeList.append(str(self.output_dir).rstrip("/") + "/" + outFile + ".v")
            return str(self.output_dir).rstrip("/") + "/" + outFile + ".v"
        else:
            return myFile
        
    def checkAll(self, myFile):
        #check for TR
        self.progress()
        if((self.ui.cb_temporal_filtering.isChecked() or self.ui.cb_movement_correction.isChecked()) and not self.ui.cb_set_repetition.isChecked()):
            if( not self.checkHeaderFor(myFile, "repetition_time") ):
                self.throwError("The repetition time in " + myFile + " is missing. Please set it manually.")
                return False
        #check for slicetime
        if( self.ui.cb_slicetime_correction.isChecked()):
            if( not self.checkHeaderFor(myFile, "slice_time") ):
                self.throwError("The slicetime information in " + myFile + " is missing. Please deselect slicetime correction.")
                return False
            
        return True

#################################################################################################################################

#Tatsaechliche Behandlungsschritte

    def doSliceTimeCorrection(self,file):
        self.progress()
        self.debugOutput("Applying slicetime correction to " + file + "...", True)
        outFile = str(self.output_dir).rstrip("/") + "/slicetime_" + file.split("/")[len(file.split("/"))-1]
        self.applyCommand("vslicetime -in " + str(file) + " -out " + str(outFile))
        self.removeList.append(outFile)
        return outFile

    def doMovementCorrection(self,file):
        self.progress()
        self.debugOutput("Applying movement correction to " + file + "...", True)
        outFile = str(self.output_dir).rstrip("/") + "/movcorr_" + file.split("/")[len(file.split("/"))-1]
        self.applyCommand("vmovcorrection -in " + str(file) + " -out " + str(outFile))
        self.removeList.append(outFile)
        return outFile
    
    def doRegistration(self, file):
        self.progress()
        outFile = str(self.output_dir).rstrip("/") + "/reg_" + file.split("/")[len(file.split("/"))-1]
        #registration
        prealign = ""
        if(not self.ui.cb_prealign_images.isChecked()):
            prealign = " -prealign false"
        bound = self.ui.sb_max_deformation.text()
        iter_bspline = self.ui.sb_max_iterations_p3.value()
        iter_affine = self.ui.sb_max_iterations_p2.value()
        iter_rigid = self.ui.sb_max_iterations_p1.value()
        transform = "0"
        optimizer = "0"
        resolution = str(self.ui.le_output_resolution.text()).replace(",", " ", 3)
        max_iterations = str(iter_rigid)
        if(self.ui.cb_affine_registration.isChecked()):
            optimizer += " 0"
            transform += " 1"
            max_iterations += " " + str(iter_affine)
        if(self.ui.cb_deformable_registration.isChecked()):
            optimizer += " 2"
            transform += " 2"
            max_iterations += " " + str(iter_bspline)
        interpolator = self.ui.comboBox_interpol_func.currentText()
        #check if an anatomical image was specified
        origFile = str(self.myFileList[self.currentImageIndex])
        movingFile = file
        anatomicalFile = ""
        ana=True
        try:
            anatomicalFile = self.anatomicalImageDict[str(origFile)]
        except:
            ana=False
            pass

        if(ana):
            self.debugOutput("Registration of anatomical image " + anatomicalFile + " to functional image " + movingFile, True)
            self.applyCommand("valign3d -ref " + movingFile + " -in " + anatomicalFile + " -trans ana_to_func.nii")
            self.debugOutput("Resampling anatomical image", True)
            self.applyCommand("vdotrans3d -ref " + movingFile + " -in " + anatomicalFile + " -trans ana_to_func.nii -res 1 -out ana_to_func.v")
            self.debugOutput("Registration of anatomical image to template " + str(self.ui.comboBox_template.currentText()), True)
            self.applyCommand("valign3d -ref " + self.templateDict[str(self.ui.comboBox_template.currentText())] + " -in ana_to_func.v -trans trans.nii -transform " + transform +" -optimizer " + optimizer + " -bound " + str(bound) + " " + prealign +  " -iter " +  str(max_iterations) + " -v")
        else:
            self.debugOutput("Registration of " + movingFile + " on " + str(self.ui.comboBox_template.currentText()) + "..." , True)
            self.applyCommand("valign3d -ref " + self.templateDict[str(self.ui.comboBox_template.currentText())] + " -in " + movingFile + " -trans trans.nii -transform " + transform + " -optimizer " + optimizer + " -bound " + str(bound) + " " + prealign + " -iter " + str(max_iterations) + " -v")
        #resampling
        if(self.ui.cb_show_registration_results.isChecked() and ana):
            self.progress()
            self.debugOutput("Resampling the image to check the registration result...", True)
            self.applyCommand("vdotrans3d -ref " + self.templateDict[str(self.ui.comboBox_template.currentText())] + " -in ana_to_func.v -trans trans.nii -out regResult.v")
            self.debugOutput("Opening viewer to show the registration results. Close it to continue!", True)
            self.applyCommand("vlv " + self.templateDict[str(self.ui.comboBox_template.currentText())] + " & ")
            self.applyCommand("vlv regResult.v")
            
        self.debugOutput("Resampling the image " + outFile + "...", True)
        self.progress() 
        self.applyCommand("vdotrans3d -ref " + self.templateDict[str(self.ui.comboBox_template.currentText())] + " -in " + file + " -trans trans.nii -res " + resolution + " -fmri -out " + outFile + " -interpolator " + str(interpolator))
        self.removeList.append(outFile)
        self.removeListEver.append("trans.nii")
        self.removeListEver.append("ana_to_func.v")
        self.removeListEver.append("ana_to_func.nii")
        self.removeListEver.append("regResult.v")
        return outFile


    def doSpatialFiltering(self,file):
        self.progress()
        outFile = str(self.output_dir).rstrip("/") + "/sfilter_" + file.split("/")[len(file.split("/"))-1]
        self.debugOutput("Applying spatial filter to " + file + "...", True)
        self.applyCommand("vpreprocess -in " + file + " -out " + outFile + " -fwhm " + str(self.ui.sb_fwhm.value()))
        self.removeList.append(outFile)
        return outFile
        
    def doTemporalFiltering(self,file):
        self.progress()
        outFile = str(self.output_dir).rstrip("/") + "/tfilter_" + file.split("/")[len(file.split("/"))-1]
        self.debugOutput("Applying temporal filter to " + file + "...", True)
        highString=lowString=""
        if(self.ui.cb_high_pass.isChecked()):
            highString=" -high " + str(self.ui.sb_hp_cutoff.value())
        if(self.ui.cb_low_pass.isChecked()):
            lowString=" -low " + str(self.ui.sb_lp_cutoff.value())
        self.applyCommand("vpreprocess -in " + file + "  -out " + outFile + highString + lowString)
        self.removeList.append(outFile)
        return outFile

    def doCreateMaskFromFile(self,file,origName):
        self.progress()
        maskFile = str(self.output_dir).rstrip("/") + "/mask_" + origName.split("/")[len(origName.split("/"))-1]
        self.debugOutput("Creating mask for image " + origName + "...", True)
        self.applyCommand("vtimestep -in " + file + " -out tmpTimeStep.v")
        self.createMask("tmpTimeStep.v", maskFile)
        
    def doCreateMaskAverage(self,imageList):
        self.progress()
        imageString = ""
        index=0
        timestepList = []
        for file in imageList:
            self.applyCommand("vtimestep -in " + file + " -out " + str(index) + ".v")
            timestepList.append(str(index) + ".v")
            self.removeListEver.append(str(index) + ".v")
            imageString += str(index) + ".v "
            index+=1
        self.applyCommand("vave -in " + imageString + " -out tmpAverage.v")
        self.removeTmpFiles()
        self.createMask("tmpAverage.v", str(self.output_dir).rstrip("/") + "/" + "mask.v")
        self.removeListEver.append("tmpAverage.v")
        
    def createMask(self,input,output):
        self.debugOutput("Creating mask...", True)
        threshold = 70
        maxvoxel = int(self.ui.sb_max_num_voxel.text())
        voxelcount = maxvoxel + 1
        self.removeListEver.append("tmpMask.v")
        self.removeListEver.append("tmpMaskSmoothed.v")
        self.removeListEver.append("tmpMaskFloat.v")
        self.removeListEver.append("volumeinfo.tmp")
        while(voxelcount > maxvoxel):
            self.applyCommand("vbinarize -in " + input + " -out tmpMask.v -min "+ str(threshold))
            self.applyCommand("vsmooth3d -in tmpMask.v -out tmpMaskSmoothed.v -iter 1000")
            self.applyCommand("vconvert -in tmpMaskSmoothed.v -out tmpMaskFloat.v -repn float")
            os.system("volumeinfo -in tmpMaskSmoothed.v 2> volumeinfo.tmp")
            volumefile = open("volumeinfo.tmp")
            lines = volumefile.readlines()
            for line in lines:
                if(string.find(line, "0:") != -1):
                    voxelcount =  float((line.split(":")[1].split(",")[1]).rstrip("\n"))
            threshold+=1
            if(threshold == 255):
                self.throwError("Creating of mask failed!")
                return False
        if(self.ui.cb_show_mask.isChecked()):
            self.applyCommand("vlv -in " + input + " -z tmpMaskFloat.v")
        self.applyCommand("mv tmpMaskSmoothed.v " + output)

################################################################################################################################## 

    def throwError(self, msg):
        QtGui.QMessageBox.critical(self, "Error", msg)

    def calculateProgressBar(self):
        maxProgress=0
        self.ui.progressBar.setMaximum(maxProgress)
        #converting + check:
        maxProgress+=len(self.myFileList)*2
        if(self.ui.cb_slicetime_correction.isChecked()):
            maxProgress+=len(self.myFileList)
        if(self.ui.cb_movement_correction.isChecked()):
            maxProgress+=len(self.myFileList)
        if(self.ui.cb_atlas_registration.isChecked()):
            maxProgress+=len(self.myFileList)*2
        if(self.ui.cb_show_registration_results.isChecked()):
            maxProgress+=len(self.myFileList)
        if(self.ui.cb_spatial_filtering.isChecked()):
            maxProgress+=len(self.myFileList)
        if(self.ui.cb_temporal_filtering.isChecked()):
            maxProgress+=len(self.myFileList)
        if(self.ui.cb_create_mask.isChecked() and self.ui.rb_create_mask_subject.isChecked()):
            maxProgress+=len(self.myFileList)
        if(self.ui.cb_create_mask.isChecked() and self.ui.rb_create_average_mask.isChecked()):
            maxProgress+=1
        self.ui.progressBar.setMaximum(maxProgress)
        

    def checkHeaderFor(self, file, attribute):
        self.debugOutput("Checking " + file + " for attribute " + attribute, True)
        os.system("less " + str(file) + " > check")
        checkFile = open("check")
        for line in checkFile.readlines():
            if(string.find(line, attribute) != -1):
                checkFile.close()
                os.remove("check")
                return True
        return False
        
    def progress(self, size=1):
        value = self.ui.progressBar.value() + size
        self.ui.progressBar.setValue(value)
        self.app.processEvents()

    def debugOutput(self, outString, statusBar):
        if(self.ui.cb_debug_output.isChecked()):
            print outString
        if(statusBar):
            self.statusBar().showMessage(outString)
            self.app.processEvents()
            
    def applyCommand(self, command):
        if (self.ui.cb_debug_output.isChecked()):
            print "-> " + command
            os.system(str(command))
        else:
            os.system(str(command) + " > tmpOut 2> tmpOut")
            
    def removeTmpFiles(self):
        for fileToRemove in self.removeListEver:
            try:
                os.remove(fileToRemove)
            except:
                pass
            
    def exit_program(self):
        self.removeTmpFiles()
        self.debugOutput("Removing temporary files. Please standby...", True)
        if(not self.ui.cb_write_logfile.isChecked()):
            for fileToRemove in self.removeList:
                try:
                    os.remove(fileToRemove)
                except:
                    pass           
        try:
            os.remove("tmpOut")
        except:
            pass
        print "Good bye!"

##################################################################################################################################

# Execute Function
if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    myapp = PreProc(app)
    myapp.show()
    sys.exit(app.exec_())
