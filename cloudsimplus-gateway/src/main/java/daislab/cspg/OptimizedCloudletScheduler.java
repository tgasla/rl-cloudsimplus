package daislab.cspg;

import org.cloudsimplus.brokers.DatacenterBroker;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.cloudlets.CloudletExecution;
import org.cloudsimplus.schedulers.MipsShare;
import org.cloudsimplus.schedulers.cloudlet.CloudletSchedulerAbstract;
import org.cloudsimplus.schedulers.cloudlet.CloudletSchedulerSpaceShared;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.Set;

class OptimizedCloudletScheduler extends CloudletSchedulerSpaceShared {

    @Override
    protected double cloudletSubmitInternal(
                final CloudletExecution cle, 
                final double fileTransferTime) {
        if (!getVm().isCreated()) {
            // It is possible, that we schedule a cloudlet, an event with processing
            // update is issued (tag: 16), but the VM gets killed before the event
            // is processed. In such a case the cloudlet does not get rescheduled,
            // because we don't know yet that this cloudlet should be!
            final Cloudlet cloudlet = cle.getCloudlet();
            final DatacenterBroker broker = cloudlet.getBroker();
            broker.submitCloudletList(Collections.singletonList(cloudlet.reset()));

            return -1.0;
        }

        return super.cloudletSubmitInternal(cle, fileTransferTime);
    }

    @Override
    public double updateProcessing(final double currentTime, final MipsShare mipsShare) {
        final int sizeBefore = getCloudletWaitingList().size();
        final double nextSimulationTime = super.updateProcessing(currentTime, mipsShare);
        final int sizeAfter = getCloudletWaitingList().size();

        // if we have a new cloudlet being processed, 
        // schedule another recalculation, which should trigger a proper
        // estimation of end time
        if (sizeAfter != sizeBefore && Double.MAX_VALUE == nextSimulationTime) {
            return getVm().getSimulation().getMinTimeBetweenEvents();
        }

        return nextSimulationTime;
    }

    @Override
    protected Optional<CloudletExecution> findSuitableWaitingCloudlet() {
        if (getVm().getProcessor().getAvailableResource() > 0) {
            final List<CloudletExecution> cloudletWaitingList = getCloudletWaitingList();
            for (CloudletExecution cle : cloudletWaitingList) {
                if (isThereEnoughFreePesForCloudlet(cle)) {
                    return Optional.of(cle);
                }
            }
        }

        return Optional.empty();
    }

    private void setPrivateField(final String fieldName, final Object value) 
            throws NoSuchFieldException, IllegalAccessException {

        final Field field = CloudletSchedulerAbstract.class.getDeclaredField(fieldName);
        field.setAccessible(true);
        field.set(this, value);
    }

    private Set<?> getPrivateFieldValueAsSet(final String fieldName, final Object source)
            throws IllegalAccessException, NoSuchFieldException {

        final Field field = CloudletSchedulerAbstract.class.getDeclaredField(fieldName);
        field.setAccessible(true);
        Object fieldValue = field.get(source);
        if (fieldValue instanceof Set) {
            return (Set<?>) fieldValue;
        } else {
            // Handle the case where the field value is not a Set appropriately
            return null;
        }
    }

    // It is safe to override this function:
    // it is used only in one place - DatacenterBrokerAbstract:827
    @Override
    public void clear() {
        try {
            setPrivateField("cloudletWaitingList", new ArrayList<>());
            setPrivateField("cloudletExecList", new ArrayList<>());

            Set<?> cloudletReturnedList = 
                    getPrivateFieldValueAsSet("cloudletReturnedList", this);
                    
            if (cloudletReturnedList != null) {
                cloudletReturnedList.clear();
            }
        } catch (NoSuchFieldException | IllegalAccessException e) {
            e.printStackTrace();
        }
    }
}
