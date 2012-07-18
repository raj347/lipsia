PROJECT(vsulcogram)

ADD_EXECUTABLE(vsulcogram CleanImage.c vsulcogram.c)
TARGET_LINK_LIBRARIES(vsulcogram lipsia ${VIA_LIBRARY})


INSTALL(TARGETS vsulcogram
        RUNTIME DESTINATION bin
        COMPONENT RuntimeLibraries)
