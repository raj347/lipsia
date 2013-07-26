PROJECT(vbrainpeel)

ADD_EXECUTABLE(vbrainpeel vbrainpeel.c vbrainpeel_impl.c LocalContrast.c)
TARGET_LINK_LIBRARIES(vbrainpeel lipsia ${VIA_LIBRARY})


INSTALL(TARGETS vbrainpeel
        RUNTIME DESTINATION bin
        COMPONENT RuntimeLibraries)
