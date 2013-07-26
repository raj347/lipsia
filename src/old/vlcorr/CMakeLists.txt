PROJECT(vlcorr)

INCLUDE_DIRECTORIES(${QT_INCLUDE_DIR} ../lib_vlh ../lib_vlh/xpm)

ADD_EXECUTABLE(vlcorr vlcorr.C)
TARGET_LINK_LIBRARIES(vlcorr ${QT_LIBRARIES} vlh lipsia ${VIA_LIBRARY})

INSTALL(TARGETS vlcorr
        RUNTIME DESTINATION bin
        COMPONENT RuntimeLibraries)
