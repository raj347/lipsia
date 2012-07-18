PROJECT(vlview)

INCLUDE_DIRECTORIES(${QT_INCLUDE_DIR} ../lib_vlh ../lib_vlh/xpm)

ADD_EXECUTABLE(vlview vlview.C)
TARGET_LINK_LIBRARIES(vlview ${QT_LIBRARIES} vlh lipsia ${VIA_LIBRARY})


INSTALL(TARGETS vlview
        RUNTIME DESTINATION bin
        COMPONENT RuntimeLibraries)
