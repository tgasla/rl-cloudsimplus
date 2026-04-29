package daislab.cspg;

import java.lang.StringBuilder;

public class TreeArray {
    final int[] state;

    public TreeArray(final int[] state) {
        this.state = state;
    }

    public String toDot() {
        StringBuilder dotString = new StringBuilder("graph {\n");
        dotString.append(createNodeLabel("d0", state[0])); // Create the root node (d0)

        int hostNumIdx = 1;
        int hostCoreIdx = hostNumIdx + 1;

        // Loop over the hosts
        for (int i = 0; i < state[hostNumIdx]; i++) {
            hostCoreIdx = appendHost(dotString, state, i, "d0", hostCoreIdx);
        }

        dotString.append("}");
        return dotString.toString();
    }

    private int appendHost(StringBuilder dotString, int[] state, int hostIndex, String parent,
            int hostCoreIdx) {
        String hostName = "h" + hostIndex;

        // Append the host node and its connection to the parent
        dotString.append(createNodeLabel(hostName, state[hostCoreIdx]));
        dotString.append(createEdge(parent, hostName));

        int vmNumIdx = hostCoreIdx + 1;
        int vmCoreIdx = vmNumIdx + 1;

        // Loop over the VMs for this host
        for (int j = 0; j < state[vmNumIdx]; j++) {
            vmCoreIdx = appendVm(dotString, state, hostName, j, vmCoreIdx);
        }

        return vmCoreIdx;
    }

    private int appendVm(StringBuilder dotString, int[] state, String hostName, int vmIndex,
            int vmCoreIdx) {
        String vmName = hostName + "v" + vmIndex;

        // Append the VM node and its connection to the host
        dotString.append(createNodeLabel(vmName, state[vmCoreIdx]));
        dotString.append(createEdge(hostName, vmName));

        int jobNumIdx = vmCoreIdx + 1;
        int jobCoreIdx = jobNumIdx + 1;

        // Loop over the jobs for this VM
        for (int k = 0; k < state[jobNumIdx]; k++) {
            jobCoreIdx = appendJob(dotString, state, vmName, k, jobCoreIdx);
        }

        return jobCoreIdx;
    }

    private int appendJob(StringBuilder dotString, int[] state, String vmName, int jobIndex,
            int jobCoreIdx) {
        String jobName = vmName + "j" + jobIndex;

        // Append the job node and its connection to the VM
        dotString.append(createNodeLabel(jobName, state[jobCoreIdx]));
        dotString.append(createEdge(vmName, jobName));

        return jobCoreIdx + 2; // Move to the next job (each job has 2 core indices)
    }

    private String createNodeLabel(String nodeName, int label) {
        return nodeName + " [label=" + label + "]\n";
    }

    private String createEdge(String parent, String child) {
        return parent + "--" + child + "\n";
    }
}
