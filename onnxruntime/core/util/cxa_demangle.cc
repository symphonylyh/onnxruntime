extern "C" {
#include <stddef.h>
#include <limits.h>

char* __cxa_demangle(const char* mangled_name, char* output_buffer, size_t* length, int* status) {
  if (!status) return NULL;
  if (!mangled_name || !output_buffer || !length || (*length == 0) || (*length > INT_MAX)) {
    *status = -3;
    return NULL;
  }

  int i = 0;
  int limit = (int)(*length) - 1;
  for (; i < limit; i++) {
    output_buffer[i] = mangled_name[i];
    if (mangled_name[i] == '\0') break;
  }

  *status = 0;
  return output_buffer;
}

}
