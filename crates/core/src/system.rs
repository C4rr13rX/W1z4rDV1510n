use tracing::{debug, warn};

#[derive(Debug)]
pub struct NumaBindingGuard {
    #[cfg(target_os = "windows")]
    process: windows_sys::Win32::Foundation::HANDLE,
    #[cfg(target_os = "windows")]
    original_mask: usize,
}

unsafe impl Send for NumaBindingGuard {}
unsafe impl Sync for NumaBindingGuard {}

impl Drop for NumaBindingGuard {
    fn drop(&mut self) {
        #[cfg(target_os = "windows")]
        unsafe {
            if self.original_mask != 0 {
                windows_sys::Win32::System::Threading::SetProcessAffinityMask(
                    self.process,
                    self.original_mask,
                );
            }
        }
    }
}

#[derive(Debug)]
pub struct LockedBuffer {
    ptr: *mut u8,
    len: usize,
    large_page: bool,
}

unsafe impl Send for LockedBuffer {}
unsafe impl Sync for LockedBuffer {}

impl LockedBuffer {
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    pub fn len(&self) -> usize {
        self.len
    }
}

impl Drop for LockedBuffer {
    fn drop(&mut self) {
        #[cfg(target_os = "windows")]
        unsafe {
            use windows_sys::Win32::System::Memory::{MEM_RELEASE, VirtualFree, VirtualUnlock};
            if self.len > 0 {
                let _ = VirtualUnlock(self.ptr as *mut _, self.len);
                let _ = VirtualFree(self.ptr as *mut _, 0, MEM_RELEASE);
            }
        }
    }
}

pub fn configure_numa_affinity(cpu_cores: usize) -> Option<NumaBindingGuard> {
    #[cfg(target_os = "windows")]
    {
        use windows_sys::Win32::Foundation::HANDLE;
        use windows_sys::Win32::System::Threading::{
            GetCurrentProcess, GetCurrentThread, GetProcessAffinityMask, SetProcessAffinityMask,
            SetThreadAffinityMask,
        };
        unsafe {
            let process: HANDLE = GetCurrentProcess();
            let thread = GetCurrentThread();
            let mut process_mask: usize = 0;
            let mut system_mask: usize = 0;
            if GetProcessAffinityMask(process, &mut process_mask, &mut system_mask) == 0 {
                warn!(
                    target: "w1z4rdv1510n::system",
                    "failed querying process affinity mask"
                );
                return None;
            }
            let mut desired_mask = 0usize;
            let mut cores_set = 0usize;
            for bit in 0..usize::BITS {
                let mask = 1usize << bit;
                if process_mask & mask != 0 {
                    desired_mask |= mask;
                    cores_set += 1;
                }
                if cores_set >= cpu_cores {
                    break;
                }
            }
            if desired_mask == 0 {
                return None;
            }
            if SetProcessAffinityMask(process, desired_mask) == 0 {
                warn!(
                    target: "w1z4rdv1510n::system",
                    "failed to update process affinity mask"
                );
                return None;
            }
            let _ = SetThreadAffinityMask(thread, desired_mask);
            debug!(
                target: "w1z4rdv1510n::system",
                mask = format_args!("{:#x}", desired_mask),
                "NUMA/affinity configured"
            );
            return Some(NumaBindingGuard {
                process,
                original_mask: process_mask,
            });
        }
    }
    #[cfg(not(target_os = "windows"))]
    {
        let _ = cpu_cores;
        None
    }
}

pub fn allocate_large_page_buffer(bytes: usize) -> Option<LockedBuffer> {
    #[cfg(target_os = "windows")]
    {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_LARGE_PAGES, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualLock,
        };
        unsafe {
            let ptr = VirtualAlloc(
                std::ptr::null_mut(),
                bytes,
                MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES,
                PAGE_READWRITE,
            ) as *mut u8;
            if ptr.is_null() {
                warn!(
                    target: "w1z4rdv1510n::system",
                    bytes,
                    "large-page allocation failed (privilege missing?)"
                );
                return None;
            }
            if VirtualLock(ptr as *mut _, bytes) == 0 {
                warn!(
                    target: "w1z4rdv1510n::system",
                    "large-page VirtualLock failed; releasing allocation"
                );
                use windows_sys::Win32::System::Memory::{MEM_RELEASE, VirtualFree};
                let _ = VirtualFree(ptr as *mut _, 0, MEM_RELEASE);
                return None;
            }
            debug!(
                target: "w1z4rdv1510n::system",
                bytes,
                "allocated large-page buffer"
            );
            return Some(LockedBuffer {
                ptr,
                len: bytes,
                large_page: true,
            });
        }
    }
    #[cfg(not(target_os = "windows"))]
    {
        let _ = bytes;
        None
    }
}

pub fn allocate_numa_buffer(bytes: usize) -> Option<LockedBuffer> {
    #[cfg(target_os = "windows")]
    {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAllocExNuma, VirtualLock,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        unsafe {
            let process = GetCurrentProcess();
            let ptr = VirtualAllocExNuma(
                process,
                std::ptr::null_mut(),
                bytes,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
                0,
            ) as *mut u8;
            if ptr.is_null() {
                warn!(
                    target: "w1z4rdv1510n::system",
                    bytes,
                    "NUMA allocation failed"
                );
                return None;
            }
            if VirtualLock(ptr as *mut _, bytes) == 0 {
                warn!(
                    target: "w1z4rdv1510n::system",
                    "VirtualLock failed for NUMA buffer"
                );
            }
            debug!(
                target: "w1z4rdv1510n::system",
                bytes,
                "allocated NUMA-local buffer"
            );
            return Some(LockedBuffer {
                ptr,
                len: bytes,
                large_page: false,
            });
        }
    }
    #[cfg(not(target_os = "windows"))]
    {
        let _ = bytes;
        None
    }
}
