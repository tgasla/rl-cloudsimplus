package daislab.cspg;

import org.cloudsimplus.brokers.DatacenterBrokerSimple;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.cloudlets.CloudletSimple;
import org.cloudsimplus.core.CloudSimPlus;
import org.cloudsimplus.core.CloudSimTag;
import org.cloudsimplus.core.events.SimEvent;
import org.cloudsimplus.datacenters.Datacenter;
import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.vms.VmSimple;

import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Collections;

/**
 * Fixed version of the original class - uses list of currently executable VMs instead of created
 * ones (what makes the cloudlets go puff)
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

    @Override
    public void processEvent(final SimEvent evt) {
        super.processEvent(evt);
        /*
         * This is important! CLOUDLET_RETURN is sent whenever a cloudlet finishes executing. The
         * default behaviour in CloudSim Plus is to destroy a Vm when it has no more cloudlets to
         * execute. Here we override the default behaviour and we trigger the creation of waiting
         * cloudlets so they can be possibly allocated inside a Vm. This is because in our case, we
         * may reschedule some cloudlets so we want the VMs to trigger the creation of those waiting
         * cloudlets
         */

        if (evt.getTag() == CloudSimTag.CLOUDLET_RETURN) {
            final Cloudlet cloudlet = (Cloudlet) evt.getData();
            LOGGER.debug("Cloudlet {} in VM {} returned. Scheduling more cloudlets...",
                    cloudlet.getId(), cloudlet.getVm().getId());
            requestDatacentersToCreateWaitingCloudlets();
        }

        // Clean the vm created list because over an episode we may create/destroy
        // many vms so we do not want to run out of memory.
        if (evt.getTag() == CloudSimTag.VM_CREATE_ACK) {
            LOGGER.debug("Cleaning the vmCreatedList");
            getVmCreatedList().clear();
        }
    }

    /*
     * This function triggers immediately the cloudlet to vm mapping and this behaviour leads to 0
     * waitingTime for all cloudlets, which is not realistic, that's why we have commented out this
     * part.
     */
    // @Override
    // protected void requestDatacentersToCreateWaitingCloudlets() {
    // final List<Cloudlet> scheduled = new LinkedList<>();
    // final List<Cloudlet> cloudletWaitingList = getCloudletWaitingList();
    // for (final Iterator<Cloudlet> it = cloudletWaitingList.iterator(); it.hasNext();) {
    // final CloudletSimple cloudlet = (CloudletSimple) it.next();
    // if (!cloudlet.getLastTriedDatacenter().equals(Datacenter.NULL)) {
    // continue;
    // }

    // // selects a VM for the given Cloudlet
    // Vm selectedVm = defaultVmMapper(cloudlet);
    // if (selectedVm == Vm.NULL) {
    // break;
    // }

    // ((VmSimple) selectedVm).removeExpectedFreePesNumber(cloudlet.getPesNumber());

    // cloudlet.setVm(selectedVm);
    // send(getDatacenter(selectedVm), cloudlet.getSubmissionDelay(),
    // CloudSimTag.CLOUDLET_SUBMIT, cloudlet);
    // cloudlet.setLastTriedDatacenter(getDatacenter(selectedVm));
    // getCloudletCreatedList().add(cloudlet);
    // scheduled.add(cloudlet);
    // it.remove();
    // }

    // LOGGER.debug("requestDatacentersToCreateWaitingCloudlets scheduled: " + scheduled.size()
    // + "/" + cloudletWaitingList.size());
    // LOGGER.debug(
    // "Events cnt before: " + getSimulation().getNumberOfFutureEvents(simEvent -> true));
    // for (Cloudlet cloudlet : scheduled) {
    // final long totalLengthInMips = cloudlet.getTotalLength();
    // final double peMips = cloudlet.getVm().getProcessor().getMips();
    // final double lengthInSeconds = totalLengthInMips / peMips;
    // final Datacenter datacenter = getDatacenter(cloudlet.getVm());
    // final double eventDelay = lengthInSeconds + 1.0;

    // LOGGER.debug("Cloudlet " + cloudlet.getId() + " scheduled. Updating in: " + eventDelay);

    // getSimulation().send(datacenter, datacenter, eventDelay,
    // CloudSimTag.VM_UPDATE_CLOUDLET_PROCESSING, null);
    // }
    // LOGGER.debug(
    // "Events cnt after: " + getSimulation().getNumberOfFutureEvents(simEvent -> true));
    // }

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
        /*
         * The original code (see https://github.com/cloudsimplus/cloudsimplus/blob/master
         * /src/main/java/org/cloudsimplus/brokers/DatacenterBrokerFirstFit.java seems to be
         * next-fit or first-fit with memory because on each iteration it starts the search at the
         * position where the last search finished. A slight detail: if in the previous search the
         * job was placed in a vm, the search on the next iteration starts from this same vm (i.e.
         * if in last search the job was placed in vm 5, the next search starts again from vm 5). If
         * no vm was found on the previous search, the search starts from the beginning.
         */

        if (cloudlet.isBoundToVm()) {
            return cloudlet.getVm();
        }

        // No VMs available
        if (getVmExecList().isEmpty()) {
            return Vm.NULL;
        }

        /**
         * If we delete a VM when in the previous iteration we had lastVmIndex set to size() - 1
         * then we are going to explode... if we don't have the line below :) For example, if the
         * lastVmIndex is 9 and then we delete a vm, the lastvmindex will cause an
         * OutOfBoundsException if we call getVmExecList().get(lastVmIndex) because now there are 9
         * vms in the vmExecList (indeces [0-8]). The line below will do lastVmIndex = 0 (9 mod 9)
         * The rest of the code remains the same as the original CloudSim Plus implementation.
         */
        lastVmIndex %= getVmExecList().size();

        /**
         * The for loop just defines the maximum number of Vms to try. When a suitable Vm is found,
         * the method returns immediately.  
         */
        final int maxTries = getVmExecList().size();
        for (int i = 0; i < maxTries; i++) {
            final Vm vm = getVmExecList().get(lastVmIndex);
            if (vm.getExpectedFreePesNumber() >= cloudlet.getPesNumber()) {
                LOGGER.trace("{}: {}: {} (PEs: {}) mapped to {} (available PEs: {}, tot PEs: {})",
                        getSimulation().clockStr(), getName(), cloudlet, cloudlet.getPesNumber(),
                        vm, vm.getExpectedFreePesNumber(), vm.getFreePesNumber());
                return vm;
            }

            /*
             * If it gets here, the previous Vm doesn't have capacity to place the Cloudlet. Then,
             * moves to the next Vm. If the end of the Vm list is reached, starts from the
             * beginning, until the max number of tries.
             */
            lastVmIndex = ++lastVmIndex % getVmExecList().size();
        }

        LOGGER.warn("{}: {}: {} (PEs: {}) couldn't be mapped to any suitable VM.",
                getSimulation().clockStr(), getName(), cloudlet, cloudlet.getPesNumber());

        // if we return NULL, that VM is not created so the cloudlet lands
        // on "waiting" list
        return Vm.NULL;
    }
}
