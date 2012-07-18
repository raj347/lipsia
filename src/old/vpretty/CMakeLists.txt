PROJECT(vpretty)

ADD_EXECUTABLE(vpretty vpretty.c)
TARGET_LINK_LIBRARIES(vpretty lipsia ${VIA_LIBRARY})


INSTALL(TARGETS vpretty
        RUNTIME DESTINATION bin
        COMPONENT RuntimeLibraries)
