PROJECT(vwilcoxon)

ADD_EXECUTABLE(vwilcoxon vwilcoxon.c)
TARGET_LINK_LIBRARIES(vwilcoxon lipsia ${VIA_LIBRARY})


INSTALL(TARGETS vwilcoxon
        RUNTIME DESTINATION bin
        COMPONENT RuntimeLibraries)
