package daislab.cspg;

import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.vms.VmSimple;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.AfterAll;

import java.io.File;
import org.apache.commons.io.FileUtils;

import static org.hamcrest.CoreMatchers.equalTo;
import static org.hamcrest.number.IsCloseTo.closeTo;
import static org.hamcrest.MatcherAssert.assertThat;

public class VmCostTest {

    final int mipsCapacity = 4400;
    private VmCost vmCost;

    @BeforeEach
    public void setUp() throws Exception {
        vmCost = new VmCost(0.2, 60, false);
    }

    @Test
    public void testCostOfNoMachinesIsZero() {
        assertThat(vmCost.getVMCostPerIteration(0), equalTo(0.0));
    }

    @Test
    public void testCostFor11VMs() {
        // S
        vmCost.addNewVmToList(createVmS());

        // 10x M
        for (int i = 0; i < 10; i++) {
            vmCost.addNewVmToList(createVmM());
        }

        assertThat(vmCost.getVMCostPerIteration(1), closeTo(0.07, 0.001));
    }

    private Vm createVmS() {
        return new VmSimple(mipsCapacity, 2).setDescription("S");
    }

    private Vm createVmM() {
        return new VmSimple(mipsCapacity, 4).setDescription("M");
    }
        
    @AfterAll
    public static void deleteJobLogDirectory() {
        // Recursively delete the tempDirectory and its contents
        try {
            FileUtils.deleteDirectory(new File("./logs"));
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}