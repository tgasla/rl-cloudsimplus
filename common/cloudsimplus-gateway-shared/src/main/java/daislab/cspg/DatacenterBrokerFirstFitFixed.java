package daislab.cspg;

import org.cloudsimplus.brokers.DatacenterBrokerSimple;
import org.cloudsimplus.core.CloudSimPlus;
import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.cloudlets.Cloudlet;

/**
 * Fixed version of the original class - uses list of currently executable VMs instead of created
 * ones (which makes the cloudlets go puff)
 */
public class DatacenterBrokerFirstFitFixed extends DatacenterBrokerSimple {
    /**
     * The index of the last Vm used to place a Cloudlet.
     */
    private int lastVmIndex;

    /**
     * Creates a DatacenterBroker object.
     *
     * @param simulation The CloudSim instance that represents the simulation the Entity is related
     *        to
     */
    public DatacenterBrokerFirstFitFixed(final CloudSimPlus simulation) {
        super(simulation);
    }

    /**
     * Here, we override the original function which tries to find a vm from the created list to
     * place a cloudlet. Instead, because we may also remove a vm, the default logic does not make
     * sense for us, so we try to find a vm for a cloudlet using the current executing vms instead
     * of the created ones.
     *
     * @param cloudlet the Cloudlet to find a VM to run it
     * @return the VM selected for the Cloudlet or {@link Vm#NULL} if no suitable VM was found
     */
    @Override
    public Vm defaultVmMapper(final Cloudlet cloudlet) {
        if (cloudlet.isBoundToVm()) {
            return cloudlet.getVm();
        }

        if (getVmExecList().isEmpty()) {
            return Vm.NULL;
        }

        lastVmIndex %= getVmExecList().size();

        final int maxTries = getVmExecList().size();
        for (int i = 0; i < maxTries; i++) {
            final Vm vm = getVmExecList().get(lastVmIndex);
            if (vm.getExpectedFreePesNumber() >= cloudlet.getPesNumber()) {
                LOGGER.trace("{}: {}: {} (PEs: {}) mapped to {} (available PEs: {}, tot PEs: {})",
                        getSimulation().clockStr(), getName(), cloudlet, cloudlet.getPesNumber(),
                        vm, vm.getExpectedFreePesNumber(), vm.getFreePesNumber());
                return vm;
            }

            lastVmIndex = ++lastVmIndex % getVmExecList().size();
        }

        LOGGER.debug("{}: {}: {} (PEs: {}) couldn't be mapped to any suitable VM.",
                getSimulation().clockStr(), getName(), cloudlet, cloudlet.getPesNumber());

        return Vm.NULL;
    }
}
