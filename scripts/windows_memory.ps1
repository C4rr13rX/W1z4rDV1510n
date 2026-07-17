if (-not ("WizardVision.NativeMemory" -as [type])) {
    Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;

namespace WizardVision
{
    public static class NativeMemory
    {
        [StructLayout(LayoutKind.Sequential)]
        public sealed class MemoryStatus
        {
            public uint Length = (uint)Marshal.SizeOf(typeof(MemoryStatus));
            public uint MemoryLoad;
            public ulong TotalPhysical;
            public ulong AvailablePhysical;
            public ulong TotalPageFile;
            public ulong AvailablePageFile;
            public ulong TotalVirtual;
            public ulong AvailableVirtual;
            public ulong AvailableExtendedVirtual;
        }

        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        public static extern bool GlobalMemoryStatusEx(
            [In, Out] MemoryStatus status
        );
    }
}
'@
}

function Get-WizardAvailableMemoryMb {
    $status = [WizardVision.NativeMemory+MemoryStatus]::new()
    if (-not [WizardVision.NativeMemory]::GlobalMemoryStatusEx($status)) {
        $errorCode = [Runtime.InteropServices.Marshal]::GetLastWin32Error()
        throw "GlobalMemoryStatusEx failed with Win32 error $errorCode."
    }
    return [double]($status.AvailablePhysical / 1MB)
}
