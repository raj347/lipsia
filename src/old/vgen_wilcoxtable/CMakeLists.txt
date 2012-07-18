PROJECT(vgen_wilcoxtable)

ADD_EXECUTABLE(vgen_wilcoxtable vgen_wilcoxtable.c)
TARGET_LINK_LIBRARIES(vgen_wilcoxtable lipsia ${VIA_LIBRARY})

INSTALL(TARGETS vgen_wilcoxtable
        RUNTIME DESTINATION bin
        COMPONENT RuntimeLibraries)
